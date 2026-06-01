"""OpenAICompatibleProvider — remote embedding via OpenAI-compatible API.

Requires the ``[embeddings-openai]`` extra (``httpx``).
Works with OpenAI, Azure OpenAI, and any compatible endpoint.

Related:
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider:
    """Remote embedding provider using the OpenAI embeddings API.

    Communicates via ``httpx`` with any OpenAI-compatible endpoint.
    Connection is lazy — no network I/O in ``__init__``.

    Args:
        model_name: Embedding model to use.
        base_url: API base URL. Defaults to OpenAI's endpoint.
        api_key: Bearer token. Falls back to ``OPENAI_API_KEY`` env var.
        dimension: Expected vector dimensionality. Auto-detected on
            first call if not provided.

    Examples:
        >>> provider = OpenAICompatibleProvider(
        ...     model_name="text-embedding-3-small",
        ...     api_key="sk-test",
        ...     dimension=1536,
        ... )
        >>> provider.model_name
        'text-embedding-3-small'

    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._dimension: int | None = dimension
        self._client: Any = None

    def _get_client(self) -> Any:  # noqa: ANN401
        """Lazy-create the httpx async client.

        Returns:
            An ``httpx.AsyncClient`` instance.

        Raises:
            ImportError: If ``httpx`` is not installed.

        """
        if self._client is not None:
            return self._client

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "httpx is required for OpenAICompatibleProvider. "
                "Install with: pip install engrava[embeddings-openai]"
            )
            raise ImportError(msg) from exc

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
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

    async def embed(self, text: str) -> list[float]:
        """Encode a single text via the OpenAI embeddings API.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            RuntimeError: On API error.

        """
        results = await self._request_embeddings([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts via the OpenAI embeddings API.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: On API error.

        """
        return await self._request_embeddings(texts)

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Send a batch embedding request to the API.

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors, ordered by input index.

        Raises:
            RuntimeError: On non-200 response or malformed JSON.

        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }

        response = await client.post("/embeddings", json=payload)
        if response.status_code != 200:  # noqa: PLR2004
            msg = f"OpenAI embeddings API error {response.status_code}: {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        embeddings_data = data.get("data", [])
        # Sort by index to ensure correct ordering
        embeddings_data.sort(key=lambda x: x["index"])
        vectors: list[list[float]] = [item["embedding"] for item in embeddings_data]

        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        return vectors
