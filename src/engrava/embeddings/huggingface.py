"""HuggingFaceProvider — remote embedding via HuggingFace Inference API.

Requires the ``[embeddings-hf]`` extra (``huggingface_hub``).

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
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("HF_TOKEN", "")
        self._dimension: int | None = dimension
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
