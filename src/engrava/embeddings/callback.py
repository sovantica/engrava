"""CallbackProvider — user-defined callable embedding provider.

Zero external dependencies. Wraps a synchronous ``(str) -> list[float]``
function as an async ``EmbeddingProviderProtocol``.

Related:
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class CallbackProvider:
    """Embedding provider backed by a user-supplied callback function.

    Wraps a synchronous callable ``(str) -> list[float]`` into the
    async ``EmbeddingProviderProtocol``.  Useful for integrating existing
    embedding pipelines without migration.

    Args:
        callback: Synchronous function that encodes text to a vector.
        dimension: Vector dimensionality.
        model_name: Identifier string for metadata tracking.

    Examples:
        >>> provider = CallbackProvider(
        ...     callback=lambda t: [0.1] * 384,
        ...     dimension=384,
        ...     model_name="dummy-384",
        ... )
        >>> provider.dimension
        384

    """

    def __init__(
        self,
        callback: Callable[[str], list[float]],
        dimension: int,
        model_name: str = "callback",
    ) -> None:
        self._callback = callback
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality.

        Returns:
            Vector dimension.

        """
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            Model identifier string.

        """
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        """Encode a single text via the wrapped callback.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.

        """
        return await asyncio.to_thread(self._callback, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts by calling the callback for each.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.

        """
        return [await self.embed(t) for t in texts]
