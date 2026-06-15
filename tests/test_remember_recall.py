"""Functional contract for the ergonomic ``remember`` / ``recall`` pair.

These two convenience methods are the smallest possible surface for storing a
string and getting relevant strings back, so an agent author never has to hand-
build a :class:`~engrava.domain.models.thought.ThoughtRecord` or call
``search_hybrid`` with the right keyword arguments for the common case.

Everything here is deterministic and network-free: query embeddings come from a
:class:`~engrava.CallbackProvider` wrapping a bag-of-words hashing embedder (the
same pattern the search-contract suite uses), so the vector arm is exercised
without loading a model or reaching the network.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import CallbackProvider, SearchConfig, SqliteEngravaCore
from engrava.domain.models.thought import ThoughtRecord
from engrava.domain.protocols.engrava_core import EngravaCoreProtocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Deterministic embedding provider (bag-of-words hashing — network-free)
# ---------------------------------------------------------------------------

_EMBED_DIM = 128


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric word tokens.

    Args:
        text: Arbitrary input text.

    Returns:
        Lowercase word tokens, with punctuation stripped.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _bag_of_words_embed(text: str) -> list[float]:
    """Embed text as an L2-normalized bag-of-words hashing vector.

    Cosine similarity between two such vectors grows with the fraction of
    shared vocabulary, giving ``recall`` a deterministic, network-free
    semantic signal.

    Args:
        text: Input text to embed.

    Returns:
        An ``_EMBED_DIM``-length unit vector (all-zero only for empty text).
    """
    vector = [0.0] * _EMBED_DIM
    for token in _tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
        index = int.from_bytes(digest[:4], "big") % _EMBED_DIM
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_provider() -> CallbackProvider:
    """Return a deterministic bag-of-words embedding provider.

    Returns:
        A :class:`CallbackProvider` wrapping the network-free hashing embedder.
    """
    return CallbackProvider(
        callback=_bag_of_words_embed,
        dimension=_EMBED_DIM,
        model_name="bag-of-words-remember-recall",
    )


async def _make_store(
    *,
    embedding_provider: CallbackProvider | None = None,
    auto_embed: bool = False,
    search_config: SearchConfig | None = None,
) -> SqliteEngravaCore:
    """Build a schema-applied in-memory store.

    Args:
        embedding_provider: Optional provider for the vector arm.
        auto_embed: Whether to auto-embed thoughts on write.
        search_config: Optional search-weight configuration.

    Returns:
        A ready-to-use :class:`SqliteEngravaCore` over an in-memory database.
    """
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    store = SqliteEngravaCore(
        conn,
        embedding_provider=embedding_provider,
        auto_embed=auto_embed,
        search_config=search_config,
    )
    await store.ensure_schema()
    return store


@pytest.fixture
async def fts_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return an empty FTS-only store (no embedding provider).

    Yields:
        A :class:`SqliteEngravaCore` over an in-memory database.
    """
    store = await _make_store()
    yield store
    await store._db.close()


@pytest.fixture
async def hybrid_store(
    embedding_provider: CallbackProvider,
) -> AsyncIterator[SqliteEngravaCore]:
    """Return an empty store with a deterministic vector arm.

    Args:
        embedding_provider: The network-free bag-of-words provider.

    Yields:
        A :class:`SqliteEngravaCore` with ``auto_embed`` enabled.
    """
    store = await _make_store(embedding_provider=embedding_provider, auto_embed=True)
    yield store
    await store._db.close()


@pytest.fixture
async def recency_store() -> AsyncIterator[SqliteEngravaCore]:
    """Return an FTS-only store configured with a positive recency weight.

    The recency arm only contributes when both a ``current_cycle`` is supplied
    *and* the resolved recency weight is non-zero, so a store that means to
    exercise recency must carry a :class:`SearchConfig` with a positive
    ``default_recency_weight`` (the packaged default).

    Yields:
        A :class:`SqliteEngravaCore` whose ``SearchConfig`` enables recency.
    """
    store = await _make_store(search_config=SearchConfig(default_recency_weight=0.1))
    yield store
    await store._db.close()


# ---------------------------------------------------------------------------
# Test 1 — remember() persists a retrievable thought from a bare string
# ---------------------------------------------------------------------------


async def test_remember_persists_retrievable_thought(fts_store: SqliteEngravaCore) -> None:
    """``remember`` turns a bare string into a stored, retrievable thought."""
    text = "The alternator on the blue sedan is failing and needs replacement."
    stored = await fts_store.remember(text)

    assert isinstance(stored, ThoughtRecord)
    assert stored.content == text
    # essence is a compact prefix capped at 200 chars (here the whole string).
    assert stored.essence == text[:200]

    fetched = await fts_store.get_thought(stored.thought_id)
    assert fetched is not None
    assert fetched.content == text


async def test_remember_truncates_essence_to_200_chars(fts_store: SqliteEngravaCore) -> None:
    """A long body yields a 200-char essence while content is preserved whole."""
    text = "alternator " * 60  # ~660 chars
    stored = await fts_store.remember(text)

    assert stored.content == text
    assert stored.essence == text[:200]
    assert len(stored.essence) == 200


# ---------------------------------------------------------------------------
# Test 2 — remember(deduplicate=True) collapses byte-identical content
# ---------------------------------------------------------------------------


async def test_remember_deduplicate_collapses_identical_content(
    fts_store: SqliteEngravaCore,
) -> None:
    """``remember(deduplicate=True)`` bumps confirmation_count, not row count."""
    text = "Standup is moved to 10am on Thursdays."
    first = await fts_store.remember(text, deduplicate=True)
    second = await fts_store.remember(text, deduplicate=True)

    assert second.thought_id == first.thought_id
    assert second.confirmation_count == first.confirmation_count + 1

    # Exactly one row persisted despite two remember() calls.
    cursor = await fts_store._db.execute("SELECT COUNT(*) FROM thought")
    row = await cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 1


async def test_remember_without_dedup_creates_two_rows(fts_store: SqliteEngravaCore) -> None:
    """The default ``deduplicate=False`` inserts a new row every call."""
    text = "Standup is moved to 10am on Thursdays."
    await fts_store.remember(text)
    await fts_store.remember(text)

    cursor = await fts_store._db.execute("SELECT COUNT(*) FROM thought")
    row = await cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 2


# ---------------------------------------------------------------------------
# Test 3 — recall() returns the relevant stored string
# ---------------------------------------------------------------------------


async def test_recall_returns_relevant_thought(hybrid_store: SqliteEngravaCore) -> None:
    """``recall`` finds the turn whose vocabulary matches the query."""
    await hybrid_store.remember(
        "My sister's golden retriever Biscuit is terrified of thunderstorms."
    )
    await hybrid_store.remember("The quarterly budget spreadsheet is over by 1200 dollars.")
    await hybrid_store.remember("We are flying to Paris in October near Montmartre.")

    results = await hybrid_store.recall("what is my sister's dog afraid of", top_k=3)

    assert results.results, "recall returned no results"
    top_id = results.results[0][0]
    top = await hybrid_store.get_thought(top_id)
    assert top is not None
    assert "Biscuit" in top.content


async def test_recall_respects_top_k(fts_store: SqliteEngravaCore) -> None:
    """``recall`` caps the number of returned results at ``top_k``."""
    for i in range(5):
        await fts_store.remember(f"Note number {i} about the office fiddle leaf fig plant.")

    results = await fts_store.recall("fiddle leaf fig plant", top_k=2)
    assert len(results.results) <= 2


# ---------------------------------------------------------------------------
# Test 4 — recall(current_cycle=...) activates the recency backend
# ---------------------------------------------------------------------------


async def test_recall_current_cycle_activates_recency(recency_store: SqliteEngravaCore) -> None:
    """Passing ``current_cycle`` wires the recency signal into the fusion.

    Recency contributes only when a ``current_cycle`` is supplied *and* the
    resolved recency weight is positive; the ``recency_store`` fixture supplies
    the weight, and ``recall`` threads the cycle through to ``search_hybrid``.
    """
    await recency_store.remember("The retro is locked in for half past noon on the calendar.")

    results = await recency_store.recall("retro calendar", current_cycle=10)
    assert "recency" in results.backends_used


async def test_recall_without_current_cycle_skips_recency(
    recency_store: SqliteEngravaCore,
) -> None:
    """Omitting ``current_cycle`` leaves the recency backend out of the fusion.

    Even with a positive recency weight configured, ``recall`` without a
    ``current_cycle`` must not activate the recency arm — confirming the cycle
    is the gating input ``recall`` forwards.
    """
    await recency_store.remember("The retro is locked in for half past noon on the calendar.")

    results = await recency_store.recall("retro calendar")
    assert "recency" not in results.backends_used


# ---------------------------------------------------------------------------
# Test 5 — remember(metadata=...) round-trips structured attributes
# ---------------------------------------------------------------------------


async def test_remember_round_trips_metadata(fts_store: SqliteEngravaCore) -> None:
    """Structured ``metadata`` survives a remember/get round trip byte-exact."""
    stored = await fts_store.remember(
        "Customer asked about the refund window.",
        metadata={"speaker": "agent", "turn_index": 7, "lang": "en"},
    )

    fetched = await fts_store.get_thought(stored.thought_id)
    assert fetched is not None
    assert fetched.metadata == {"speaker": "agent", "turn_index": 7, "lang": "en"}


async def test_remember_defaults_metadata_to_empty(fts_store: SqliteEngravaCore) -> None:
    """Omitting ``metadata`` stores an empty dict, not ``None``."""
    stored = await fts_store.remember("A note with no metadata.")
    fetched = await fts_store.get_thought(stored.thought_id)
    assert fetched is not None
    assert fetched.metadata == {}


# ---------------------------------------------------------------------------
# Test 6 — protocol parity (amended): methods on core + protocol, dedup wiring
# ---------------------------------------------------------------------------


def test_core_satisfies_protocol_with_remember_recall() -> None:
    """``remember`` / ``recall`` are declared on the protocol and present on core."""
    # The methods are declared on the core protocol itself.
    assert hasattr(EngravaCoreProtocol, "remember")
    assert hasattr(EngravaCoreProtocol, "recall")

    # They are concretely present on the SQLite core implementation.
    assert callable(SqliteEngravaCore.remember)
    assert callable(SqliteEngravaCore.recall)


async def test_core_instance_is_runtime_checkable_protocol_member(
    fts_store: SqliteEngravaCore,
) -> None:
    """A live ``SqliteEngravaCore`` satisfies the runtime-checkable protocol."""
    assert isinstance(fts_store, EngravaCoreProtocol)


async def test_remember_dedup_true_delegates_to_create_thought(
    fts_store: SqliteEngravaCore,
) -> None:
    """``remember(deduplicate=True)`` forwards the flag to ``create_thought``.

    Two checks together prove the flag is threaded through rather than dropped:

    * The first (outer) ``create_thought`` invocation made by ``remember``
      carries ``deduplicate=True`` — captured by a spy that records only the
      caller-facing call, not the internal re-delegation the dedup branch makes
      on a cache miss.
    * Behaviourally, a second deduplicated remember lands on the *same* row
      with a bumped ``confirmation_count`` — the observable effect of
      ``create_thought(deduplicate=True)``.
    """
    outer_calls: list[bool] = []
    original = fts_store.create_thought

    async def _spy(thought: ThoughtRecord, **kwargs: object) -> ThoughtRecord:
        outer_calls.append(bool(kwargs.get("deduplicate", False)))
        return await original(thought, **kwargs)

    fts_store.create_thought = _spy  # type: ignore[method-assign]
    first = await fts_store.remember("dedup wiring probe", deduplicate=True)
    second = await fts_store.remember("dedup wiring probe", deduplicate=True)

    # The first thing remember() calls is create_thought(deduplicate=True).
    assert outer_calls[0] is True
    # And the dedup behaviour is observable: same row, bumped confirmation.
    assert second.thought_id == first.thought_id
    assert second.confirmation_count == first.confirmation_count + 1


# ---------------------------------------------------------------------------
# Test 7 — one-time recency-off DEBUG nudge fires once per store instance
# ---------------------------------------------------------------------------


async def test_recall_emits_recency_nudge_once_over_threshold(
    fts_store: SqliteEngravaCore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A large store recalled without ``current_cycle`` emits one DEBUG nudge."""
    # Cross the threshold: more than 25 thoughts and no current_cycle passed.
    for i in range(30):
        await fts_store.remember(f"Recency nudge corpus item {i} about widgets and gadgets.")

    logger_name = "engrava.infrastructure.sqlite.engrava_core"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        await fts_store.recall("widgets gadgets")
        await fts_store.recall("widgets gadgets")
        await fts_store.recall("widgets gadgets")

    nudges = [
        record
        for record in caplog.records
        if record.name == logger_name
        and record.levelno == logging.DEBUG
        and "recency" in record.getMessage().lower()
    ]
    assert len(nudges) == 1, f"expected exactly one recency nudge, got {len(nudges)}"


async def test_recall_no_nudge_under_threshold(
    fts_store: SqliteEngravaCore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A small store does not emit the recency nudge."""
    for i in range(3):
        await fts_store.remember(f"Small corpus item {i}.")

    logger_name = "engrava.infrastructure.sqlite.engrava_core"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        await fts_store.recall("corpus item")

    nudges = [
        record
        for record in caplog.records
        if record.name == logger_name
        and record.levelno == logging.DEBUG
        and "recency" in record.getMessage().lower()
    ]
    assert nudges == []


async def test_recall_no_nudge_when_current_cycle_supplied(
    fts_store: SqliteEngravaCore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Supplying ``current_cycle`` suppresses the recency nudge even when large."""
    for i in range(30):
        await fts_store.remember(f"Nudge-suppression corpus item {i} about widgets.")

    logger_name = "engrava.infrastructure.sqlite.engrava_core"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        await fts_store.recall("widgets", current_cycle=5)

    nudges = [
        record
        for record in caplog.records
        if record.name == logger_name
        and record.levelno == logging.DEBUG
        and "recency" in record.getMessage().lower()
    ]
    assert nudges == []
