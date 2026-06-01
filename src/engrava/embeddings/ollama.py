"""OllamaProvider — local/remote embedding via Ollama API.

Requires the ``[embeddings-ollama]`` extra (``httpx``).

Related:
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """Embedding provider using the Ollama ``/api/embed`` endpoint.

    Communicates via ``httpx``. Connection is lazy — no network I/O
    in ``__init__``.

    Args:
        model_name: Ollama model name (e.g. ``"nomic-embed-text"``).
        base_url: Ollama server URL. Defaults to ``http://localhost:11434``.
        dimension: Expected vector dimensionality. Auto-detected on first call.

    Examples:
        >>> provider = OllamaProvider(dimension=768)
        >>> provider.model_name
        'nomic-embed-text'

    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = _DEFAULT_BASE_URL,
        dimension: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
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
                "httpx is required for OllamaProvider. "
                "Install with: pip install engrava[embeddings-ollama]"
            )
            raise ImportError(msg) from exc

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,
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
        """Encode a single text via the Ollama embed API.

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
        """Encode multiple texts via the Ollama embed API.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: On API error.

        """
        return await self._request_embeddings(texts)

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Send an embedding request to the Ollama API.

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: On non-200 response or malformed JSON.

        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }

        response = await client.post("/api/embed", json=payload)
        if response.status_code != 200:  # noqa: PLR2004
            msg = f"Ollama embed API error {response.status_code}: {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        vectors: list[list[float]] = data.get("embeddings", [])

        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        return vectors
