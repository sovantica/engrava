"""SentenceTransformerProvider — local embedding via sentence-transformers.

Requires the ``[embeddings-local]`` extra (``sentence-transformers``, ``torch``).
Model is lazy-loaded on first ``embed()`` call.

Related:
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SentenceTransformerProvider:
    """Local embedding provider using ``sentence-transformers``.

    The model is loaded lazily on first use to avoid heavyweight imports at
    construction time. Encoding is offloaded to a thread via
    ``asyncio.to_thread()`` since ``sentence-transformers`` is synchronous.

    Args:
        model_name: HuggingFace model identifier.
        device: Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
        batch_size: Batch encoding size for ``embed_batch()``.

    Examples:
        >>> provider = SentenceTransformerProvider()
        >>> provider.model_name
        'all-MiniLM-L12-v2'

    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self) -> Any:  # noqa: ANN401
        """Lazy-load the SentenceTransformer model.

        Returns:
            The loaded ``SentenceTransformer`` instance.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.

        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install with: pip install engrava[embeddings-local]"
            )
            raise ImportError(msg) from exc

        logger.info("Loading SentenceTransformer model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality.

        Lazy-loads the model if dimension is not yet known.

        Returns:
            Vector dimension.

        """
        if self._dimension is None:
            self._load_model()
        return self._dimension  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            Model identifier string.

        """
        return self._model_name

    def _encode_sync(self, text: str) -> list[float]:
        """Encode a single text synchronously.

        Args:
            text: Input text.

        Returns:
            Embedding vector.

        """
        model = self._load_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()  # type: ignore[no-any-return]

    def _encode_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts synchronously.

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors.

        """
        model = self._load_model()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=self._batch_size)
        return [v.tolist() for v in vecs]

    async def embed(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalized embedding vector.

        """
        return await asyncio.to_thread(self._encode_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts into embedding vectors.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.

        """
        return await asyncio.to_thread(self._encode_batch_sync, texts)
