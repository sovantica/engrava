"""OpenAICompatibleProvider — remote embedding via OpenAI-compatible API.

Requires the ``[embeddings-openai]`` extra (``httpx``).
Works with OpenAI, Azure OpenAI, and any compatible endpoint.

Related:
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: Number of attempts (initial try + retries) for a transient failure.
#: The default of 3 leaves the success path at a single request while
#: absorbing a couple of consecutive transient blips.
_DEFAULT_MAX_ATTEMPTS = 3

#: Base delay (seconds) for the exponential backoff between retries.
_DEFAULT_BASE_RETRY_DELAY_S = 1.0

#: HTTP status codes treated as transient and therefore retryable:
#: request timeout, conflict, too-early, rate limit, and the standard
#: 5xx server / gateway errors. Any other non-2xx status is surfaced
#: immediately without a retry.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_HTTP_OK = 200


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
        max_attempts: Maximum number of attempts (initial request plus
            retries) for a single embeddings call when the endpoint
            returns a transient error. Keyword-only; defaults to ``3``.
            A value of ``1`` disables retrying.
        base_retry_delay_s: Base delay in seconds for the exponential
            backoff between retries (the n-th retry waits
            ``base_retry_delay_s * n`` seconds). Keyword-only; defaults
            to ``1.0``.

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
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        base_retry_delay_s: float = _DEFAULT_BASE_RETRY_DELAY_S,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._dimension: int | None = dimension
        self._client: Any = None
        self._max_attempts = max(1, max_attempts)
        self._base_retry_delay_s = max(0.0, base_retry_delay_s)

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

        The request is retried with bounded exponential backoff when the
        endpoint reports a transient failure — a transport-level timeout
        or network error, or a transient HTTP status (see
        :data:`_RETRYABLE_STATUS_CODES`). Non-transient HTTP errors (for
        example ``400``/``401``/``404``) are surfaced immediately without
        a retry, and a transient failure that persists across every
        attempt is re-raised so the call still fails rather than looping
        forever. The number of attempts and the backoff base are
        configured on the provider (``max_attempts`` /
        ``base_retry_delay_s``).

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors, ordered by input index.

        Raises:
            RuntimeError: On a non-retryable response, or on a transient
                failure that persists across every attempt.

        """
        import httpx  # noqa: PLC0415

        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await client.post("/embeddings", json=payload)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                # Transport-level transient error (read timeout, connection
                # reset, …). The exception text carries the request target,
                # never the request headers, so no credential is exposed.
                if attempt >= self._max_attempts:
                    msg = (
                        f"OpenAI embeddings API request failed after "
                        f"{attempt} attempt(s): {type(exc).__name__}"
                    )
                    raise RuntimeError(msg) from exc
                await self._sleep_before_retry(attempt)
                continue

            if response.status_code == _HTTP_OK:
                return self._parse_response(response)

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_attempts:
                await self._sleep_before_retry(attempt)
                continue

            # Non-retryable status, or the final attempt of a retryable one:
            # surface the error. The message carries the status code and the
            # response body only — never the request's authorization header.
            msg = f"OpenAI embeddings API error {response.status_code}: {response.text}"
            raise RuntimeError(msg)

        # Unreachable: the loop either returns or raises on every path. Present
        # so the function provably never falls through without a value.
        msg = "OpenAI embeddings API request exhausted all attempts"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    async def _sleep_before_retry(self, attempt: int) -> None:
        """Sleep with exponential backoff before the next retry attempt.

        Args:
            attempt: The 1-based number of the attempt that just failed.
                The delay scales linearly with this value
                (``base_retry_delay_s * attempt``).

        """
        delay = self._base_retry_delay_s * attempt
        if delay > 0:
            await asyncio.sleep(delay)

    def _parse_response(self, response: Any) -> list[list[float]]:  # noqa: ANN401
        """Parse a successful embeddings response into ordered vectors.

        Args:
            response: The ``httpx.Response`` from a ``200`` reply.

        Returns:
            List of embedding vectors, ordered by input index.

        """
        data = response.json()
        embeddings_data = data.get("data", [])
        # Sort by index to ensure correct ordering
        embeddings_data.sort(key=lambda x: x["index"])
        vectors: list[list[float]] = [item["embedding"] for item in embeddings_data]

        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        return vectors
