"""EmbeddingProviderProtocol — async embedding generation interface.

Async-first design: all methods are ``async def``. Sync providers
(e.g. SentenceTransformer) wrap internally via ``asyncio.to_thread()``.
Never the reverse — a sync protocol would force ``asyncio.run()`` on callers.

Related:
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProviderProtocol(Protocol):
    """Abstract interface for text-to-vector embedding generation.

    All methods are async. Implementations that call synchronous libraries
    (e.g. ``sentence-transformers``) must use ``asyncio.to_thread()``
    internally.

    Attributes:
        dimension: Dimensionality of produced embedding vectors.
        model_name: Model identifier string (persisted in ``_metadata``).

    """

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality (e.g. 384).

        Returns:
            Vector dimension.

        """
        ...

    @property
    def model_name(self) -> str:
        """Return the embedding model name (e.g. 'all-MiniLM-L12-v2').

        Returns:
            Model identifier string.

        """
        ...

    async def embed(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalized embedding vector as a list of floats.

        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts into embedding vectors.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        """
        ...
