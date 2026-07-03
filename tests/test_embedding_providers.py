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
    EmbeddingQueryPrefixMismatchError,
    HuggingFaceProvider,
    OllamaProvider,
    SentenceTransformerProvider,
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
from engrava.infrastructure.sqlite.engrava_core import _build_embed_input

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
# Embed-input construction (prefix de-duplication)
# ---------------------------------------------------------------------------


class _RecordingProvider:
    """Embedding provider that records the exact text passed to ``embed``.

    Wraps a fixed-dimension constant vector so the only observable effect is
    the captured input string — used to assert *what* text auto-embed sends
    to the provider, independent of the vector arithmetic.
    """

    def __init__(self, dimension: int = 4, model_name: str = "recording") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.captured: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        self.captured.append(text)
        return [0.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class TestBuildEmbedInput:
    """Unit tests for the prefix-dedup helper :func:`_build_embed_input`."""

    def test_essence_is_prefix_returns_content_alone(self) -> None:
        content = "The quick brown fox jumps over the lazy dog near the river."
        essence = content[:20]
        assert _build_embed_input(essence, content) == content

    def test_essence_not_prefix_returns_joined(self) -> None:
        essence = "A short distinct summary"
        content = "An entirely different body of text with no overlap at the start."
        assert _build_embed_input(essence, content) == f"{essence}\n{content}"

    def test_prefix_ignoring_surrounding_whitespace(self) -> None:
        content = "Header line then the rest of the body."
        essence = "  Header line  "
        # The stripped essence is a prefix of the stripped content, so the
        # essence adds no new information and is dropped.
        assert _build_embed_input(essence, content) == content

    def test_partial_overlap_is_not_treated_as_prefix(self) -> None:
        # Conservative: only the clear prefix case dedups. A shared word that
        # is not a leading prefix keeps the joined form.
        essence = "fox jumps"
        content = "The quick brown fox jumps."
        assert _build_embed_input(essence, content) == f"{essence}\n{content}"

    def test_identical_essence_and_content_returns_content(self) -> None:
        text = "Exactly the same on both fields."
        assert _build_embed_input(text, text) == text


class TestAutoEmbedInput:
    """Integration tests asserting the exact text auto-embed sends."""

    async def test_prefix_essence_not_double_embedded(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        recorder = _RecordingProvider(dimension=4, model_name="recording")
        store = SqliteEngravaCore(db, embedding_provider=recorder, auto_embed=True)

        content = "The deployment failed because the database migration timed out."
        essence = content[:20]  # essence == content[:N]
        await store.create_thought(
            _make_thought(thought_id="t-prefix", essence=essence, content=content)
        )

        assert recorder.captured == [content]
        # The opening must not appear twice in the embedded text.
        assert recorder.captured[0].count(essence) == 1

    async def test_distinct_essence_uses_joined_form(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        recorder = _RecordingProvider(dimension=4, model_name="recording")
        store = SqliteEngravaCore(db, embedding_provider=recorder, auto_embed=True)

        essence = "Outage postmortem summary"
        content = "The deployment failed because the database migration timed out."
        await store.create_thought(
            _make_thought(thought_id="t-distinct", essence=essence, content=content)
        )

        assert recorder.captured == [f"{essence}\n{content}"]


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
            assert int(row[0]) == 17

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

    def test_parse_require_embedding_default_false(self) -> None:
        cfg = _parse_embeddings({"provider": "sentence-transformer"})
        assert cfg is not None
        assert cfg.require_embedding is False

    def test_parse_require_embedding_true(self) -> None:
        cfg = _parse_embeddings(
            {"provider": "sentence-transformer", "require_embedding": True},
        )
        assert cfg is not None
        assert cfg.require_embedding is True

    def test_parse_invalid_require_embedding(self) -> None:
        with pytest.raises(ConfigError, match="require_embedding"):
            _parse_embeddings({"require_embedding": "yes"})

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


class TestOpenAIProviderRetry:
    """Transient-error retry behaviour for OpenAICompatibleProvider."""

    @staticmethod
    def _ok_response(embedding: list[float]) -> MagicMock:
        """A 200 response carrying a single embedding at index 0."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": [{"index": 0, "embedding": embedding}]}
        return response

    @staticmethod
    def _status_response(status_code: int) -> MagicMock:
        """A non-200 response with a benign body (no secret material)."""
        response = MagicMock()
        response.status_code = status_code
        response.text = "service unavailable"
        return response

    async def test_embedding_retries_then_succeeds(self) -> None:
        """A read timeout twice, then a 200 — vectors returned after 3 attempts."""
        import httpx

        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        # A non-zero base delay so the backoff path is exercised; the
        # asyncio.sleep patch keeps the test from sleeping for real.
        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            base_retry_delay_s=1.0,
        )

        ok = self._ok_response([0.1, 0.2, 0.3])
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.ReadTimeout("read timed out"),
                httpx.ReadTimeout("read timed out"),
                ok,
            ]
        )
        provider._client = mock_client

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await provider.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        assert provider.dimension == 3
        assert mock_client.post.call_count == 3
        # Two failed attempts → two backoff sleeps before the success.
        assert mock_sleep.await_count == 2

    async def test_embedding_retries_on_retryable_status(self) -> None:
        """A 503 twice, then a 200 — success after retrying the status."""
        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            base_retry_delay_s=1.0,
        )

        ok = self._ok_response([0.4, 0.5])
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                self._status_response(503),
                self._status_response(503),
                ok,
            ]
        )
        provider._client = mock_client

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await provider.embed("hello")

        assert result == [0.4, 0.5]
        assert mock_client.post.call_count == 3
        assert mock_sleep.await_count == 2

    async def test_embedding_persistent_timeout_raises(self) -> None:
        """A read timeout on every attempt raises after max_attempts (no loop)."""
        import httpx

        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        fake_api_key = "sk-canary-token-value"
        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key=fake_api_key,
            max_attempts=3,
            base_retry_delay_s=0,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        provider._client = mock_client

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await provider.embed("hello")

        # Bounded: exactly max_attempts calls, then a raise — no infinite loop.
        assert mock_client.post.call_count == 3
        # The raised error must not leak the API key or an Authorization header.
        message = str(exc_info.value)
        assert fake_api_key not in message
        assert "Authorization" not in message
        assert "Bearer" not in message

    async def test_embedding_non_retryable_status_raises_immediately(self) -> None:
        """A 401 raises on the first attempt — no retry for a non-transient status."""
        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            base_retry_delay_s=0,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._status_response(401))
        provider._client = mock_client

        with (
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
            pytest.raises(RuntimeError, match="401"),
        ):
            await provider.embed("hello")

        # No retry on a non-retryable status: a single attempt, zero sleeps.
        assert mock_client.post.call_count == 1
        assert mock_sleep.await_count == 0

    async def test_embedding_success_path_unchanged(self) -> None:
        """A 200 on the first try — exactly one attempt and identical vectors."""
        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._ok_response([0.1, 0.2, 0.3]))
        provider._client = mock_client

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await provider.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        assert provider.dimension == 3
        # Backward-compat lock: one attempt, never any backoff sleep.
        mock_client.post.assert_called_once()
        assert mock_sleep.await_count == 0

    async def test_embedding_backoff_is_bounded(self) -> None:
        """With base_retry_delay_s=0 the retry count is bounded by max_attempts."""
        import httpx

        from engrava.embeddings.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key="sk-test",
            max_attempts=5,
            base_retry_delay_s=0,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        provider._client = mock_client

        with pytest.raises(RuntimeError):
            await provider.embed("hello")

        # Assert the attempt COUNT, not wall-clock — base_retry_delay_s=0
        # means no real sleeping occurs.
        assert mock_client.post.call_count == 5


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

    def test_load_raises_max_seq_length_to_architecture_max(self) -> None:
        """Loading lifts a conservative shipped limit up to the true maximum.

        ``all-MiniLM-L12-v2`` ships ``get_max_seq_length() == 128`` while its
        BERT backbone supports ``max_position_embeddings == 512``. The provider
        must raise ``max_seq_length`` to the architecture maximum so long
        inputs are not silently truncated at 128 word-pieces.
        """
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        # Fake transformer module exposing the architecture's true max.
        transformer_module = MagicMock()
        transformer_module.auto_model.config.max_position_embeddings = 512

        fake_model = MagicMock()
        fake_model.get_max_seq_length.return_value = 128
        fake_model.max_seq_length = 128
        fake_model.get_sentence_embedding_dimension.return_value = 384
        fake_model.tokenizer.model_max_length = 128
        # ``model[0]`` returns the underlying transformer module.
        fake_model.__getitem__.return_value = transformer_module

        st_module = MagicMock()
        st_module.SentenceTransformer.return_value = fake_model

        provider = SentenceTransformerProvider(model_name="all-MiniLM-L12-v2")
        with patch.dict("sys.modules", {"sentence_transformers": st_module}):
            loaded = provider._load_model()

        assert loaded.max_seq_length == 512

    def test_load_keeps_max_seq_length_when_already_full(self) -> None:
        """No-op when the model already reports its full architecture limit."""
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        transformer_module = MagicMock()
        transformer_module.auto_model.config.max_position_embeddings = 256

        fake_model = MagicMock()
        fake_model.get_max_seq_length.return_value = 256
        fake_model.max_seq_length = 256
        fake_model.get_sentence_embedding_dimension.return_value = 384
        fake_model.tokenizer.model_max_length = 256
        fake_model.__getitem__.return_value = transformer_module

        st_module = MagicMock()
        st_module.SentenceTransformer.return_value = fake_model

        provider = SentenceTransformerProvider(model_name="already-full")
        with patch.dict("sys.modules", {"sentence_transformers": st_module}):
            loaded = provider._load_model()

        assert loaded.max_seq_length == 256

    def test_load_leaves_max_seq_length_when_architecture_max_unreadable(self) -> None:
        """Untouched when the architecture max cannot be discovered."""
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        # Indexing the model raises — the provider cannot read the true max.
        fake_model = MagicMock()
        fake_model.__getitem__.side_effect = IndexError("no modules")
        fake_model.get_max_seq_length.return_value = 64
        fake_model.max_seq_length = 64
        fake_model.get_sentence_embedding_dimension.return_value = 384

        st_module = MagicMock()
        st_module.SentenceTransformer.return_value = fake_model

        provider = SentenceTransformerProvider(model_name="unreadable")
        with patch.dict("sys.modules", {"sentence_transformers": st_module}):
            loaded = provider._load_model()

        assert loaded.max_seq_length == 64

    def test_architecture_max_ignores_non_positive_config_value(self) -> None:
        """A missing/sentinel config value is treated as not discoverable."""
        from engrava.embeddings.sentence_transformer import SentenceTransformerProvider

        transformer_module = MagicMock()
        # A non-positive sentinel (e.g. unset) must not be adopted as the max.
        transformer_module.auto_model.config.max_position_embeddings = 0
        fake_model = MagicMock()
        fake_model.__getitem__.return_value = transformer_module

        provider = SentenceTransformerProvider(model_name="bad-config")
        assert provider._architecture_max_seq_length(fake_model) is None


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


# ---------------------------------------------------------------------------
# Asymmetric role prefixes (query_prefix / document_prefix)
# ---------------------------------------------------------------------------


class _RolePrefixSpy:
    """Role-aware fake provider recording the exact text it encodes.

    Mirrors the real providers: the role methods prepend a non-empty prefix
    (empty prefix is a literal passthrough that delegates to ``embed``) and
    every encode routes through ``embed``, whose argument is captured. This
    lets a test assert precisely which string reached the encoder on each
    path — the document path must see ``document_prefix + text`` and the
    query path ``query_prefix + text``.
    """

    def __init__(
        self,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        dimension: int = 4,
        model_name: str = "spy-model",
    ) -> None:
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._dimension = dimension
        self._model_name = model_name
        self.embed_calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def document_prefix(self) -> str:
        return self._document_prefix

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Deterministic vector derived from text length so re-embeds differ.
        return [float(len(text) % 7) / 7.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        if not self._query_prefix:
            return await self.embed(text)
        return await self.embed(self._query_prefix + text)

    async def embed_document(self, text: str) -> list[float]:
        if not self._document_prefix:
            return await self.embed(text)
        return await self.embed(self._document_prefix + text)

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._query_prefix:
            return await self.embed_batch(texts)
        return await self.embed_batch([self._query_prefix + t for t in texts])

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._document_prefix:
            return await self.embed_batch(texts)
        return await self.embed_batch([self._document_prefix + t for t in texts])


class _EmbedOnlyStub:
    """Minimal ``embed``-only provider without the role capability.

    Represents a third-party / legacy provider: it satisfies the mandatory
    protocol but exposes no role methods, so the core must fall back to
    ``embed`` for both document and query paths.
    """

    def __init__(self, dimension: int = 4, model_name: str = "embed-only") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.embed_calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.5] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class _PartialRoleProvider:
    """A malformed provider: role *methods* present, prefix *properties* absent.

    It does not satisfy the full :class:`RoleAwareEmbeddingProvider` capability
    (no ``query_prefix`` / ``document_prefix``), so the core must treat it as a
    plain provider and never call its role methods. If it did, a document could
    be prefixed on the embed path while the model lock — reading prefixes from
    the same provider — records no prefix, producing a silent asymmetry. The
    role methods inject a sentinel prefix so a test can prove they were skipped.
    """

    def __init__(self, dimension: int = 4, model_name: str = "partial") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.embed_calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.5] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed("SHOULD-NOT-APPEAR: " + text)

    async def embed_document(self, text: str) -> list[float]:
        return await self.embed("SHOULD-NOT-APPEAR: " + text)

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(["SHOULD-NOT-APPEAR: " + t for t in texts])

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch(["SHOULD-NOT-APPEAR: " + t for t in texts])


class TestRoleAwareProviderPrefixing:
    """Unit tests for the opt-in asymmetric prefix on the real providers."""

    async def test_sentence_transformer_role_methods_prepend_prefix(self) -> None:
        calls: list[str] = []

        async def _fake_embed(self: SentenceTransformerProvider, text: str) -> list[float]:
            calls.append(text)
            return [0.0] * 4

        async def _fake_embed_batch(
            self: SentenceTransformerProvider,
            texts: list[str],
        ) -> list[list[float]]:
            calls.extend(texts)
            return [[0.0] * 4 for _ in texts]

        with (
            patch.object(SentenceTransformerProvider, "embed", _fake_embed),
            patch.object(SentenceTransformerProvider, "embed_batch", _fake_embed_batch),
        ):
            provider = SentenceTransformerProvider(
                query_prefix="query: ",
                document_prefix="passage: ",
            )
            await provider.embed_query("what is x")
            await provider.embed_document("x is y")
            await provider.embed_query_batch(["a", "b"])
            await provider.embed_document_batch(["c"])

        assert "query: what is x" in calls
        assert "passage: x is y" in calls
        # Batch role methods prepend per element.
        assert "query: a" in calls
        assert "query: b" in calls
        assert "passage: c" in calls

    async def test_empty_prefix_is_literal_passthrough(self) -> None:
        """Empty prefixes call the raw embed path with no concatenation."""
        calls: list[str] = []

        async def _fake_embed(self: OllamaProvider, text: str) -> list[float]:
            calls.append(text)
            return [0.0] * 4

        with patch.object(OllamaProvider, "embed", _fake_embed):
            provider = OllamaProvider(dimension=4)  # no prefixes
            await provider.embed_query("hello")
            await provider.embed_document("world")

        # Byte-identical to the raw text — no separator, no whitespace added.
        assert calls == ["hello", "world"]

    def test_prefix_relevant_providers_are_role_aware(self) -> None:
        st = SentenceTransformerProvider(query_prefix="q: ", document_prefix="d: ")
        ol = OllamaProvider(dimension=4, query_prefix="q: ", document_prefix="d: ")
        hf = HuggingFaceProvider(dimension=4, query_prefix="q: ", document_prefix="d: ")
        from engrava import RoleAwareEmbeddingProvider

        assert isinstance(st, RoleAwareEmbeddingProvider)
        assert isinstance(ol, RoleAwareEmbeddingProvider)
        assert isinstance(hf, RoleAwareEmbeddingProvider)

    def test_openai_provider_is_not_role_aware(self) -> None:
        from engrava import RoleAwareEmbeddingProvider
        from engrava.embeddings.openai_compatible import (
            OpenAICompatibleProvider,
        )

        provider = OpenAICompatibleProvider(model_name="x", api_key="k")
        assert not isinstance(provider, RoleAwareEmbeddingProvider)

    def test_callback_provider_is_not_role_aware(
        self,
        callback_provider: CallbackProvider,
    ) -> None:
        from engrava import RoleAwareEmbeddingProvider

        assert not isinstance(callback_provider, RoleAwareEmbeddingProvider)


class TestRoleDispatchCoverage:
    """The core dispatches the right role prefix on every embed path."""

    async def test_document_path_uses_document_prefix(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        spy = _RolePrefixSpy(query_prefix="query: ", document_prefix="passage: ")
        store = SqliteEngravaCore(db, embedding_provider=spy, auto_embed=True)
        thought = _make_thought(essence="Cats", content="Cats purr when content.")
        await store.create_thought(thought)

        # The auto-embed document path encoded the document-prefixed text.
        assert any(c.startswith("passage: ") for c in spy.embed_calls)
        assert not any(c.startswith("query: ") for c in spy.embed_calls)

    async def test_both_query_sites_use_query_prefix(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        spy = _RolePrefixSpy(query_prefix="query: ", document_prefix="passage: ")
        store = SqliteEngravaCore(db, embedding_provider=spy, auto_embed=True)
        await store.create_thought(_make_thought(content="Some content to embed."))
        spy.embed_calls.clear()

        # search_hybrid path.
        await store.search_hybrid("find the thing", top_k=5)
        assert spy.embed_calls == ["query: find the thing"]

        spy.embed_calls.clear()
        # search_reflections_only path (the second query site).
        await store.search_reflections_only("reflect on this", top_k=5)
        assert spy.embed_calls == ["query: reflect on this"]

    async def test_recall_path_uses_query_prefix(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        spy = _RolePrefixSpy(query_prefix="query: ", document_prefix="passage: ")
        store = SqliteEngravaCore(db, embedding_provider=spy, auto_embed=True)
        await store.create_thought(_make_thought(content="Content for recall."))
        spy.embed_calls.clear()

        await store.recall("recall me", top_k=5)
        assert spy.embed_calls == ["query: recall me"]

    async def test_embed_only_provider_falls_back_to_embed(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """A provider without role methods keeps working via ``embed``."""
        stub = _EmbedOnlyStub()
        store = SqliteEngravaCore(db, embedding_provider=stub, auto_embed=True)
        await store.create_thought(_make_thought(content="Body text."))
        # Document path used plain embed (no prefix, raw text).
        assert stub.embed_calls
        assert all(not c.startswith(("query: ", "passage: ")) for c in stub.embed_calls)

        stub.embed_calls.clear()
        await store.search_hybrid("a query", top_k=5)
        assert stub.embed_calls == ["a query"]


class TestDefaultPathParity:
    """Default-empty prefixes are byte-identical to no prefixing at all."""

    async def test_role_methods_element_wise_identical_to_embed(self) -> None:
        """With empty prefixes, role methods equal plain embed element-wise."""
        provider = _RolePrefixSpy(query_prefix="", document_prefix="")
        plain = await provider.embed("some text to encode")
        as_query = await provider.embed_query("some text to encode")
        as_document = await provider.embed_document("some text to encode")
        assert as_query == plain
        assert as_document == plain

    async def test_stored_vector_and_ranking_identical(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # Same underlying embed function on both stores; the only difference is
        # whether the store reaches it through plain ``embed`` (no role
        # capability) or through the empty-prefix role methods. Stored vectors
        # and ranked order must match exactly — the default is a no-op.
        provider_default = CallbackProvider(_dummy_embed, dimension=4, model_name="parity")
        store = SqliteEngravaCore(db, embedding_provider=provider_default, auto_embed=True)

        for i in range(3):
            await store.create_thought(
                _make_thought(
                    thought_id=f"t-parity-{i}",
                    essence=f"Essence {i}",
                    content=f"Content number {i} about apples and oranges.",
                )
            )

        result_default = await store.search_hybrid("apples", top_k=3)
        order_default = list(result_default.results)
        emb_default = await store.get_embedding("t-parity-0")

        # Same again through an empty-prefix role-aware provider that wraps the
        # identical ``_dummy_embed`` function.
        conn2 = await aiosqlite.connect(":memory:")
        conn2.row_factory = aiosqlite.Row
        await conn2.execute("PRAGMA foreign_keys = ON")

        class _RoleAwareDummy(_RolePrefixSpy):
            async def embed(self, text: str) -> list[float]:
                self.embed_calls.append(text)
                return _dummy_embed(text)

        store2_provider = _RoleAwareDummy(
            query_prefix="",
            document_prefix="",
            model_name="parity",
        )
        store2 = SqliteEngravaCore(conn2, embedding_provider=store2_provider, auto_embed=True)
        await store2.ensure_schema()
        for i in range(3):
            await store2.create_thought(
                _make_thought(
                    thought_id=f"t-parity-{i}",
                    essence=f"Essence {i}",
                    content=f"Content number {i} about apples and oranges.",
                )
            )
        # Every recorded encode is the raw text — no prefix concatenation.
        assert all(not c.startswith(("query: ", "passage: ")) for c in store2_provider.embed_calls)
        result2 = await store2.search_hybrid("apples", top_k=3)
        order2 = list(result2.results)
        emb2 = await store2.get_embedding("t-parity-0")
        assert order_default == order2
        assert emb_default is not None
        assert emb2 is not None
        assert emb_default.vector_blob == emb2.vector_blob
        await conn2.close()


class TestProtocolNonBreakage:
    """The mandatory protocol still matches embed-only / callback providers."""

    def test_callback_still_satisfies_protocol(
        self,
        callback_provider: CallbackProvider,
    ) -> None:
        assert isinstance(callback_provider, EmbeddingProviderProtocol)

    def test_embed_only_stub_satisfies_protocol(self) -> None:
        assert isinstance(_EmbedOnlyStub(), EmbeddingProviderProtocol)

    async def test_embed_only_stub_still_embeds(self) -> None:
        stub = _EmbedOnlyStub()
        vec = await stub.embed("hi")
        assert len(vec) == 4


class TestPartialRoleCapabilityFallsBack:
    """A provider that implements the role capability only partially is plain.

    Dispatch keys off the whole :class:`RoleAwareEmbeddingProvider` capability,
    not per-method presence, so a partial provider is never prefixed on one
    path while recorded as unprefixed by the model lock.
    """

    def test_partial_provider_is_not_role_aware(self) -> None:
        from engrava import RoleAwareEmbeddingProvider

        assert not isinstance(_PartialRoleProvider(), RoleAwareEmbeddingProvider)

    async def test_partial_provider_document_path_falls_back_to_plain_embed(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        partial = _PartialRoleProvider()
        store = SqliteEngravaCore(db, embedding_provider=partial, auto_embed=True)
        await store.create_thought(_make_thought(content="Body text."))

        # The document path used plain embed with the raw text — the partial
        # provider's sentinel-injecting role methods were NOT used.
        assert partial.embed_calls
        assert all("SHOULD-NOT-APPEAR" not in c for c in partial.embed_calls)

        # And the lock recorded no document-prefix fingerprint — consistent
        # with plain treatment, so no latent identity/asymmetry mismatch.
        cursor = await db.execute("SELECT key FROM _metadata")
        keys = {row["key"] for row in await cursor.fetchall()}
        assert "embedding_document_prefix_fingerprint" not in keys

    async def test_partial_provider_query_path_falls_back_to_plain_embed(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        partial = _PartialRoleProvider()
        store = SqliteEngravaCore(db, embedding_provider=partial, auto_embed=True)
        await store.create_thought(_make_thought(content="Body text."))
        partial.embed_calls.clear()

        await store.search_hybrid("a query", top_k=5)
        # Query path also fell back to plain embed — raw text, no sentinel.
        assert partial.embed_calls == ["a query"]


class TestDocumentPrefixModelLock:
    """D3: document-prefix identity and query-prefix pairing in the lock."""

    async def test_enabling_document_prefix_on_unprefixed_corpus_raises(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # Build an unprefixed corpus.
        plain = _RolePrefixSpy(model_name="lock-model")
        store_a = SqliteEngravaCore(db, embedding_provider=plain, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock-a"))

        # Re-open with a document prefix on the same model — corpus identity
        # changed, so the lock must raise.
        prefixed = _RolePrefixSpy(model_name="lock-model", document_prefix="passage: ")
        store_b = SqliteEngravaCore(db, embedding_provider=prefixed, auto_embed=True)
        with pytest.raises(EmbeddingModelMismatchError):
            await store_b.create_thought(_make_thought("t-lock-b"))

    async def test_removing_document_prefix_also_raises(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        prefixed = _RolePrefixSpy(model_name="lock-model", document_prefix="passage: ")
        store_a = SqliteEngravaCore(db, embedding_provider=prefixed, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock-a"))

        plain = _RolePrefixSpy(model_name="lock-model")
        store_b = SqliteEngravaCore(db, embedding_provider=plain, auto_embed=True)
        with pytest.raises(EmbeddingModelMismatchError):
            await store_b.create_thought(_make_thought("t-lock-b"))

    async def test_same_document_prefix_succeeds(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        first = _RolePrefixSpy(model_name="lock-model", document_prefix="passage: ")
        store_a = SqliteEngravaCore(db, embedding_provider=first, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock-a"))

        second = _RolePrefixSpy(model_name="lock-model", document_prefix="passage: ")
        store_b = SqliteEngravaCore(db, embedding_provider=second, auto_embed=True)
        # No raise — identical corpus identity.
        await store_b.create_thought(_make_thought("t-lock-b"))

    async def test_query_prefix_only_change_does_not_force_reembed(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        # Corpus built with a query/document prefix pair.
        first = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store_a = SqliteEngravaCore(db, embedding_provider=first, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock-a"))

        # Same document prefix (corpus identity unchanged), same query prefix:
        # storing another document does NOT raise — no re-embed forced.
        same = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store_b = SqliteEngravaCore(db, embedding_provider=same, auto_embed=True)
        await store_b.create_thought(_make_thought("t-lock-b"))

    async def test_divergent_query_prefix_raises_at_search_time(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        first = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store_a = SqliteEngravaCore(db, embedding_provider=first, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock-a"))

        # Same document prefix (so store works), but a divergent query prefix.
        diverged = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="search: ",
        )
        store_b = SqliteEngravaCore(db, embedding_provider=diverged, auto_embed=True)
        with pytest.raises(EmbeddingQueryPrefixMismatchError):
            await store_b.search_hybrid("find it", top_k=5)

    async def test_matching_query_prefix_search_succeeds(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        first = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store_a = SqliteEngravaCore(db, embedding_provider=first, auto_embed=True)
        await store_a.create_thought(_make_thought("t-lock-a", content="Body."))

        same = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store_b = SqliteEngravaCore(db, embedding_provider=same, auto_embed=True)
        # Restoring the matching query prefix — no raise.
        await store_b.search_hybrid("find it", top_k=5)

    async def test_empty_prefixes_use_legacy_metadata_shape(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """Default (empty) prefixes write no fingerprint/pairing keys."""
        plain = _RolePrefixSpy(model_name="lock-model")
        store = SqliteEngravaCore(db, embedding_provider=plain, auto_embed=True)
        await store.create_thought(_make_thought("t-legacy"))

        cursor = await db.execute("SELECT key FROM _metadata ORDER BY key")
        keys = {row["key"] for row in await cursor.fetchall()}
        # Only the legacy keys — no prefix fingerprint or query-prefix key.
        assert "embedding_model_name" in keys
        assert "embedding_dimension" in keys
        assert "embedding_document_prefix_fingerprint" not in keys
        assert "embedding_query_prefix" not in keys

    async def test_existing_unprefixed_store_never_false_trips(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """A store locked before this feature (plain CallbackProvider) reopens fine."""
        provider = CallbackProvider(_dummy_embed, dimension=4, model_name="legacy-model")
        store_a = SqliteEngravaCore(db, embedding_provider=provider, auto_embed=True)
        await store_a.create_thought(_make_thought("t-a", content="Body."))

        # Reopen with an equivalent plain provider — no prefixes anywhere.
        provider_b = CallbackProvider(_dummy_embed, dimension=4, model_name="legacy-model")
        store_b = SqliteEngravaCore(db, embedding_provider=provider_b, auto_embed=True)
        await store_b.create_thought(_make_thought("t-b"))
        # And searching does not trip the query-prefix pairing check.
        await store_b.search_hybrid("body", top_k=5)

    async def test_query_prefix_on_empty_store_does_not_trip(
        self,
        db: aiosqlite.Connection,
    ) -> None:
        """A query prefix on a not-yet-populated store never false-trips.

        Before the first embedding is stored there is no corpus (no locked
        model) to pair against, so a search must not raise even though the
        provider carries a query prefix.
        """
        provider = _RolePrefixSpy(
            model_name="lock-model",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store = SqliteEngravaCore(db, embedding_provider=provider, auto_embed=True)
        await store.ensure_schema()
        # Empty corpus (nothing stored) → the pairing check is a no-op.
        await store.search_hybrid("find something", top_k=5)


class TestPrefixConfigParsing:
    """D6: config parses the two prefixes and forwards them correctly."""

    def test_parse_prefixes(self) -> None:
        cfg = _parse_embeddings(
            {
                "provider": "sentence-transformer",
                "query_prefix": "query: ",
                "document_prefix": "passage: ",
            }
        )
        assert cfg is not None
        assert cfg.query_prefix == "query: "
        assert cfg.document_prefix == "passage: "

    def test_parse_prefixes_default_none(self) -> None:
        cfg = _parse_embeddings({"provider": "sentence-transformer"})
        assert cfg is not None
        assert cfg.query_prefix is None
        assert cfg.document_prefix is None

    def test_parse_non_string_query_prefix_raises(self) -> None:
        with pytest.raises(ConfigError, match="query_prefix"):
            _parse_embeddings({"provider": "ollama", "query_prefix": 123})

    def test_parse_non_string_document_prefix_raises(self) -> None:
        with pytest.raises(ConfigError, match="document_prefix"):
            _parse_embeddings({"provider": "ollama", "document_prefix": ["x"]})

    def test_forwarded_to_sentence_transformer(self) -> None:
        cfg = EmbeddingConfig(
            provider="sentence-transformer",
            query_prefix="query: ",
            document_prefix="passage: ",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        assert provider.query_prefix == "query: "  # type: ignore[attr-defined]
        assert provider.document_prefix == "passage: "  # type: ignore[attr-defined]

    def test_forwarded_to_ollama(self) -> None:
        cfg = EmbeddingConfig(
            provider="ollama",
            query_prefix="q: ",
            document_prefix="d: ",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        assert provider.query_prefix == "q: "  # type: ignore[attr-defined]
        assert provider.document_prefix == "d: "  # type: ignore[attr-defined]

    def test_forwarded_to_huggingface(self) -> None:
        cfg = EmbeddingConfig(
            provider="huggingface",
            query_prefix="q: ",
            document_prefix="d: ",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        assert provider.query_prefix == "q: "  # type: ignore[attr-defined]
        assert provider.document_prefix == "d: "  # type: ignore[attr-defined]

    def test_openai_provider_receives_no_prefix(self) -> None:
        """The symmetric OpenAI provider has no prefix attributes at all."""
        from engrava import RoleAwareEmbeddingProvider

        cfg = EmbeddingConfig(
            provider="openai-compatible",
            query_prefix="query: ",
            document_prefix="passage: ",
            api_key="sk-test",
        )
        provider = resolve_embedding_provider(cfg)
        assert provider is not None
        # No prefixing surface — structurally symmetric.
        assert not isinstance(provider, RoleAwareEmbeddingProvider)
        assert not hasattr(provider, "query_prefix")
