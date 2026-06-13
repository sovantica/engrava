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
        self._raise_max_seq_length(self._model)
        self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @staticmethod
    def _architecture_max_seq_length(model: Any) -> int | None:  # noqa: ANN401
        """Read the underlying transformer's true maximum sequence length.

        Inspects the first pipeline module's ``auto_model.config`` for
        ``max_position_embeddings`` — the number of position embeddings the
        architecture was trained with, i.e. the longest input it can encode
        without an index error. Returns ``None`` when the value cannot be read
        (non-standard module layout), in which case the caller leaves the
        model's own limit untouched.

        Args:
            model: The loaded ``SentenceTransformer`` instance.

        Returns:
            The architecture's maximum sequence length, or ``None`` if it is
            not discoverable.

        """
        try:
            transformer_module = model[0]
            value = transformer_module.auto_model.config.max_position_embeddings
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _raise_max_seq_length(self, model: Any) -> None:  # noqa: ANN401
        """Lift a conservatively-low ``max_seq_length`` to the architecture max.

        Some ``sentence-transformers`` checkpoints ship a ``max_seq_length``
        far below the limit their backbone supports — the default
        ``all-MiniLM-L12-v2`` reports ``128`` even though its BERT backbone has
        ``max_position_embeddings == 512``. Left unchanged, the encoder
        silently truncates any input past the shipped limit, so the tail of a
        long thought is invisible to vector search and recall quietly degrades.

        This reads the architecture's true maximum and raises
        ``model.max_seq_length`` to it only when the current value is strictly
        lower. The number is derived from the model — never hard-coded — so a
        model that already reports its full limit is a no-op, and a model whose
        architecture max cannot be read is left exactly as loaded.

        Args:
            model: The loaded ``SentenceTransformer`` instance to adjust.

        """
        architecture_max = self._architecture_max_seq_length(model)
        if architecture_max is None:
            return
        current = model.get_max_seq_length()
        if current is None or current < architecture_max:
            model.max_seq_length = architecture_max
            logger.info(
                "Raised max_seq_length for %s from %s to architecture max %d "
                "to avoid silent input truncation",
                self._model_name,
                current,
                architecture_max,
            )

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
