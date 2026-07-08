"""HuggingFaceProvider — remote embedding via HuggingFace Inference API.

Requires the ``[embeddings-hf]`` extra (``huggingface_hub``).

Optional asymmetric role prefixes (``query_prefix`` / ``document_prefix``)
support instruction-tuned models (E5, BGE, GTE). A configured prefix is
prepended to the text before feature extraction; a long document whose text
plus prefix exceeds the model's ``max_seq_length`` truncates exactly as an
unprefixed over-length input would — no special reservation is made for the
prefix.

Related:
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Embedding provider using the HuggingFace Inference API.

    Uses ``huggingface_hub.InferenceClient`` for feature-extraction.
    The client is synchronous, so calls are wrapped via
    ``asyncio.to_thread()``.

    Args:
        model_name: HuggingFace model identifier.
        api_key: HuggingFace API token. Falls back to ``HF_TOKEN`` env var.
        dimension: Expected vector dimensionality. Auto-detected on first call.
        query_prefix: Optional instruction prefix prepended to a search
            query before encoding (e.g. ``"query: "``). Keyword-only. Empty
            by default — an empty prefix is a literal passthrough, so the
            role-aware path is byte-identical to the plain ``embed`` path.
            Only a non-empty prefix is prepended.
        document_prefix: Optional instruction prefix prepended to a stored
            document before encoding (e.g. ``"passage: "``). Keyword-only.
            Empty by default, with the same passthrough guarantee.

    Examples:
        >>> provider = HuggingFaceProvider(dimension=384)
        >>> provider.model_name
        'sentence-transformers/all-MiniLM-L12-v2'

    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        api_key: str | None = None,
        dimension: int | None = None,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("HF_TOKEN", "")
        self._dimension: int | None = dimension
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._client: Any = None

    def _get_client(self) -> Any:  # noqa: ANN401
        """Lazy-create the HuggingFace InferenceClient.

        Returns:
            A ``huggingface_hub.InferenceClient`` instance.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.

        """
        if self._client is not None:
            return self._client

        try:
            from huggingface_hub import InferenceClient  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "huggingface_hub is required for HuggingFaceProvider. "
                "Install with: pip install engrava[embeddings-hf]"
            )
            raise ImportError(msg) from exc

        self._client = InferenceClient(token=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality.

        Returns:
            Vector dimension.

        Raises:
            RuntimeError: If dimension was not set and no embeddings
                have been generated yet.

        """
        if self._dimension is None:
            msg = (
                "Dimension not yet known. Either set dimension in the constructor "
                "or call embed() first."
            )
            raise RuntimeError(msg)
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            Model identifier string.

        """
        return self._model_name

    @property
    def query_prefix(self) -> str:
        """Return the query-role prefix (empty string when disabled)."""
        return self._query_prefix

    @property
    def document_prefix(self) -> str:
        """Return the document-role prefix (empty string when disabled)."""
        return self._document_prefix

    def _embed_sync(self, text: str) -> list[float]:
        """Encode a single text synchronously.

        Args:
            text: Input text.

        Returns:
            Embedding vector.

        """
        client = self._get_client()
        result = client.feature_extraction(text, model=self._model_name)
        # feature_extraction returns numpy array or list
        vec: list[float] = result.tolist() if hasattr(result, "tolist") else list(result)
        # Handle nested arrays (some models return [[...]])
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        if self._dimension is None:
            self._dimension = len(vec)
        return vec

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts synchronously.

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors.

        """
        return [self._embed_sync(t) for t in texts]

    async def embed(self, text: str) -> list[float]:
        """Encode a single text via the HuggingFace Inference API.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.

        """
        return await asyncio.to_thread(self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts via the HuggingFace Inference API.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.

        """
        return await asyncio.to_thread(self._embed_batch_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        """Encode a search query, applying the query-role prefix.

        With an empty ``query_prefix`` (the default) this delegates to the
        plain :meth:`embed` path — byte-identical output, no concatenation.

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector as a list of floats.

        """
        if not self._query_prefix:
            return await self.embed(text)
        return await self.embed(self._query_prefix + text)

    async def embed_document(self, text: str) -> list[float]:
        """Encode a stored document, applying the document-role prefix.

        With an empty ``document_prefix`` (the default) this delegates to the
        plain :meth:`embed` path — byte-identical output, no concatenation.

        Args:
            text: The document text to embed.

        Returns:
            Embedding vector as a list of floats.

        """
        if not self._document_prefix:
            return await self.embed(text)
        return await self.embed(self._document_prefix + text)

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple queries, applying the query-role prefix to each.

        Args:
            texts: The query texts to embed.

        Returns:
            List of embedding vectors.

        """
        if not self._query_prefix:
            return await self.embed_batch(texts)
        return await self.embed_batch([self._query_prefix + t for t in texts])

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple documents, applying the document-role prefix.

        Args:
            texts: The document texts to embed.

        Returns:
            List of embedding vectors.

        """
        if not self._document_prefix:
            return await self.embed_batch(texts)
        return await self.embed_batch([self._document_prefix + t for t in texts])
