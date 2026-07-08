"""EmbeddingProviderProtocol — async embedding generation interface.

Async-first design: all methods are ``async def``. Sync providers
(e.g. SentenceTransformer) wrap internally via ``asyncio.to_thread()``.
Never the reverse — a sync protocol would force ``asyncio.run()`` on callers.

The mandatory :class:`EmbeddingProviderProtocol` is deliberately minimal
(``embed``/``embed_batch`` plus ``dimension``/``model_name``). Role-aware
prefixing is exposed as a *separate, optional* capability,
:class:`RoleAwareEmbeddingProvider`, so a plain ``embed``-only provider (a
user callback, a third-party class, the symmetric OpenAI provider) keeps
satisfying the required protocol unchanged and the core falls back to
``embed`` for it.

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


@runtime_checkable
class RoleAwareEmbeddingProvider(Protocol):
    """Optional capability for providers with asymmetric role prefixes.

    Instruction-tuned embedding models (E5, BGE, GTE, Ollama
    ``nomic-embed-text``) are trained with mandatory role instructions —
    typically ``"query: "`` on a search query and ``"passage: "`` on a
    stored document — and encode noticeably worse when run without them.
    This protocol lets such a provider embed text *in a role*: the core
    calls :meth:`embed_document` on the document-embed path and
    :meth:`embed_query` on the query path.

    This capability is intentionally **separate** from the mandatory
    :class:`EmbeddingProviderProtocol`. A provider that does not implement
    these methods (a user callback, a third-party class, the symmetric
    OpenAI provider) is still a valid embedding provider; the core detects
    the absence of the role methods and falls back to plain ``embed`` /
    ``embed_batch``, so its behaviour is byte-identical to before.

    Implementations must guarantee that an *empty* prefix is a literal
    passthrough: with no configured prefix, the role methods produce output
    byte-identical to ``embed`` / ``embed_batch`` (no separator, no
    whitespace, no concatenation added).

    Attributes:
        query_prefix: String prepended to a search query before encoding.
            Empty string disables query prefixing.
        document_prefix: String prepended to a stored document before
            encoding. Empty string disables document prefixing.

    """

    @property
    def query_prefix(self) -> str:
        """Return the prefix prepended to a query before encoding.

        Returns:
            The query-role prefix, or ``""`` when disabled.

        """
        ...

    @property
    def document_prefix(self) -> str:
        """Return the prefix prepended to a document before encoding.

        Returns:
            The document-role prefix, or ``""`` when disabled.

        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Encode a search query, applying the query-role prefix.

        Args:
            text: The query text to embed.

        Returns:
            L2-normalized embedding vector as a list of floats.

        """
        ...

    async def embed_document(self, text: str) -> list[float]:
        """Encode a stored document, applying the document-role prefix.

        Args:
            text: The document text to embed.

        Returns:
            L2-normalized embedding vector as a list of floats.

        """
        ...

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple queries, applying the query-role prefix to each.

        Args:
            texts: The query texts to embed.

        Returns:
            List of embedding vectors, one per input query.

        """
        ...

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple documents, applying the document-role prefix.

        Args:
            texts: The document texts to embed.

        Returns:
            List of embedding vectors, one per input document.

        """
        ...
