"""Content-safety guard for the auto-embed-failure ``WARNING``.

An auto-embed provider failure is logged with the thought id and the
exception's *type name*, not the exception object. A provider whose error
message quotes its input (a payload-too-large or validation error, for
instance) would otherwise write the thought's own content straight into the
log.

Two independent call sites route through the same handler,
``_on_auto_embed_failure``, and both are driven here so that patching one but
not the other still fails this suite:

* the single-thought path, reached from ``create_thought`` (and
  ``update_thought``, which shares the same helper);
* the batch path, ``_batch_embed_thoughts``, reached from ``bulk_store``.

Each is checked under both the default mode (the provider's own exception
propagates unchanged) and ``require_embedding=True`` (it is normalised into
:class:`~engrava.domain.exceptions.EmbeddingGenerationError`), since both
modes reach the same logging call before they diverge on what they raise.

This suite checks three independent things about every record an engrava
logger emits while a failing call runs:

* **Shape (exact pin, not a shape match).** Every record from an engrava
  logger must carry ``record.msg`` identical to the literal format string
  passed to ``logger.warning`` (copied from the source below, not
  reconstructed), its level must be ``WARNING``, and its rendering must
  satisfy ``record.getMessage() == record.msg % record.args`` -- confirming
  the record's own message is exactly the pinned format with these args. The
  two substituted args
  are then drawn from closed sets built from the call under test rather than
  matched by shape: the thought id must equal the one this call actually
  used, and the exception type name must equal ``type(...).__name__`` of the
  exact exception instance the provider raised for this call -- not a
  hard-coded literal, so a type name with extra content glued on (e.g.
  ``f"{type(exc).__name__}: {digest}"``) is rejected too. This is checked
  with ``caplog`` capturing *every* level (``NOTSET`` on both the root logger
  and the ``engrava`` package logger, not just ``WARNING``), so an extra
  ``logger.debug(...)``/``logger.info(...)`` leak would be seen too.
* **Exactly one record.** Each failing call must emit exactly one engrava
  record -- the warning itself, at any level. Removing the warning drops
  this to zero; a second record derived from the exception (a digest at
  ``DEBUG``, for instance) raises it above one. Both fail this assertion,
  so it also proves the warning was not simply deleted.
* **Marker absence**, kept as a second, independent check: the literal
  marker text must not appear in any record's message, args, formatted
  exception, or other attributes -- catching a naive ``%s``/``str``
  interpolation or an ``extra=`` leak directly, redundantly with the shape
  check above.

Separately, the raised exception itself is asserted unchanged: the exact
provider exception instance propagates by identity under the default mode,
and under ``require_embedding=True`` the resulting
:class:`~engrava.domain.exceptions.EmbeddingGenerationError` carries that
same instance as its ``__cause__``, so a caller who wants the detail still
receives it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from engrava import (
    CallbackProvider,
    CoreThoughtRecord,
    EmbeddingGenerationError,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtType,
    ThoughtVisibility,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# A marker with no plausible overlap with this suite's own vocabulary or any
# log-format token engrava itself might emit.
_MARKER = "embed7q2vwk9fjd41cnotamemoryword"

# The exact format string ``_on_auto_embed_failure`` passes to
# ``logger.warning``, copied verbatim from
# ``src/engrava/infrastructure/sqlite/engrava_core.py`` -- not reconstructed
# or matched by shape, so something merely similar cannot satisfy it.
_ALLOWED_MESSAGE_FORMAT = (
    "Auto-embed failed for thought %s: %s. The embedding was not "
    "produced. Whether the thought row survives, and in what "
    "state, depends on the call that raised this and its "
    "surrounding transaction — see docs/api-reference.md for the "
    "specific outcomes."
)

# A record built the same way ``logging`` builds every real one, used only to
# read off which attribute names are part of the standard ``LogRecord`` shape
# -- so the allowlist below is derived from the real class, not retyped by
# hand and left to drift from it.
_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None))
)

# Attributes a *formatter* legitimately adds on top of the standard shape,
# even though nothing in this suite formats a record to text itself --
# ``caplog`` formats every record it captures as a side effect. Unlike an
# ``extra={...}`` attribute, these are formatter artifacts, never a
# logging-call payload, so they are allowed here.
_FORMATTER_SET_ATTRS: frozenset[str] = frozenset({"message", "asctime"})

_ALLOWED_RECORD_ATTRS: frozenset[str] = _STANDARD_LOG_RECORD_ATTRS | _FORMATTER_SET_ATTRS


class _MarkerQuotingProvider(CallbackProvider):
    """A ``CallbackProvider`` whose callback always raises, quoting its input.

    Every raised exception is also appended to :attr:`raised`, in the order
    it was constructed, so a test can assert identity/message-preservation
    of the *exact* instance that propagated through the store -- not merely
    something ``==`` to it.
    """

    def __init__(self, *, exception_type: type[Exception] = ValueError) -> None:
        self.raised: list[Exception] = []
        self._exception_type = exception_type
        super().__init__(callback=self._raise, dimension=4, model_name="marker-quoting")

    def _raise(self, text: str) -> list[float]:
        """Raise, quoting *text* verbatim in the exception message.

        Args:
            text: The exact payload the store asked this provider to embed.

        Raises:
            Exception: An instance of :attr:`_exception_type` quoting *text*.

        """
        exc = self._exception_type(f"embedding provider rejected input: {text!r}")
        self.raised.append(exc)
        raise exc


def _thought(thought_id: str, marker: str) -> CoreThoughtRecord:
    """Build a thought whose embed payload contains *marker*.

    Args:
        thought_id: Id to give the thought.
        marker: Unique content marker to embed in the thought's body.

    Returns:
        A thought record ready for ``create_thought`` / ``bulk_store``.

    """
    essence = "auto-embed failure content-safety probe"
    content = f"{essence}\n{marker}"
    return CoreThoughtRecord(
        thought_id=thought_id,
        thought_type=ThoughtType.OBSERVATION,
        essence=essence,
        content=content,
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="test-suite",
        confidence=0.9,
        source_type=KnowledgeSource.EXPERIENCE,
        visibility=ThoughtVisibility.SELECTIVE,
        metadata={},
    )


@pytest.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    """Fresh in-memory SQLite with the core schema bootstrapped."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    yield conn
    await conn.close()


async def _embedding_store(
    conn: aiosqlite.Connection,
    provider: _MarkerQuotingProvider,
    *,
    require_embedding: bool = False,
) -> SqliteEngravaCore:
    """Build an auto-embed store bound to *conn*, wired to *provider*."""
    store = SqliteEngravaCore(
        conn,
        embedding_provider=provider,
        auto_embed=True,
        require_embedding=require_embedding,
    )
    await store._probe_fts()
    return store


def _engrava_records(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    """Return only the records emitted by an ``engrava``-namespaced logger."""
    return [record for record in records if record.name.startswith("engrava")]


def _assert_matches_allowed_content_free_shape(
    records: list[logging.LogRecord],
    *,
    allowed_thought_ids: frozenset[str],
    allowed_exception_type_names: frozenset[str],
) -> None:
    """Assert every record is exactly the one pinned, content-free shape.

    Args:
        records: Records to check (typically pre-filtered to an engrava
            logger via :func:`_engrava_records`).
        allowed_thought_ids: The thought id(s) this call legitimately named.
        allowed_exception_type_names: The exception type name(s) this call's
            provider legitimately raised.

    Raises:
        AssertionError: If any record deviates from the shape above.

    """
    for record in records:
        assert record.levelno == logging.WARNING, (
            f"expected the record at WARNING, got {record.levelname}"
        )
        assert record.msg == _ALLOWED_MESSAGE_FORMAT, (
            f"record format string is not the one the fix emits: {record.msg!r}"
        )

        args = record.args
        assert isinstance(args, tuple), f"expected a plain args tuple, got {args!r}"
        assert len(args) == 2, (
            f"expected exactly 2 args (thought id, exception type name), got {args!r}"
        )
        thought_id, exc_type_name = args

        rendered = record.msg % args
        assert record.getMessage() == rendered, (
            f"rendered message does not match the pinned format string and args: "
            f"{record.getMessage()!r} != {rendered!r}"
        )

        assert isinstance(thought_id, str), f"expected the thought id as a str, got {thought_id!r}"
        assert thought_id in allowed_thought_ids, (
            f"{thought_id!r} is not one of the thought ids this call actually used: "
            f"{sorted(allowed_thought_ids)}"
        )

        assert isinstance(exc_type_name, str), (
            f"expected the exception type name as a str, got {exc_type_name!r}"
        )
        assert exc_type_name in allowed_exception_type_names, (
            f"{exc_type_name!r} is not one of this call's own exception type names: "
            f"{sorted(allowed_exception_type_names)}"
        )

        assert record.exc_info is None, f"expected no exc_info attached, got {record.exc_info!r}"
        assert record.stack_info is None, (
            f"expected no stack_info attached, got {record.stack_info!r}"
        )

        extra_attrs = set(vars(record)) - _ALLOWED_RECORD_ATTRS
        assert not extra_attrs, (
            f"record carries attribute(s) beyond a standard LogRecord "
            f"(likely an extra= payload): {sorted(extra_attrs)} in {vars(record)!r}"
        )


def _assert_marker_absent_from_every_record(
    records: list[logging.LogRecord],
    marker: str,
) -> None:
    """Assert *marker* is absent from every record's message, args, and attributes.

    Args:
        records: Captured log records to check, at any level.
        marker: The unique content marker that must never appear.

    Raises:
        AssertionError: If any record carries the marker on any surface.

    """
    for record in records:
        message = record.getMessage()
        assert marker not in message, f"marker leaked via log message: {message!r}"
        args_text = str(record.args)
        assert marker not in args_text, f"marker leaked via log args: {args_text!r}"
        record_dict_text = repr(vars(record))
        assert marker not in record_dict_text, (
            f"marker leaked via a record attribute (e.g. extra=): {record_dict_text!r}"
        )


def _widen_caplog(caplog: pytest.LogCaptureFixture) -> None:
    """Capture every level from both the root and the ``engrava`` logger.

    Args:
        caplog: The fixture to widen.

    """
    caplog.set_level(logging.NOTSET)
    caplog.set_level(logging.NOTSET, logger="engrava")


# ---------------------------------------------------------------------------
# Single-thought path (create_thought -> _auto_embed_thought)
# ---------------------------------------------------------------------------


class TestSingleThoughtPathNeverLogsProviderMessage:
    """``create_thought``'s auto-embed failure never logs the provider's message."""

    async def test_default_mode(
        self,
        db: aiosqlite.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Default mode: the WARNING is content-free; the provider's own exception propagates."""
        provider = _MarkerQuotingProvider()
        store = await _embedding_store(db, provider)
        thought_id = "t-single-default"
        _widen_caplog(caplog)

        with pytest.raises(ValueError) as exc_info:
            await store.create_thought(_thought(thought_id, _MARKER))

        assert len(provider.raised) == 1, "expected exactly one provider call to have raised"
        original_exc = provider.raised[0]
        assert exc_info.value is original_exc, (
            "the default path must propagate the provider's own exception by identity"
        )

        engrava_records = _engrava_records(caplog.records)
        assert len(engrava_records) == 1, (
            f"expected exactly one engrava record (the warning), got {len(engrava_records)}: "
            f"{[r.getMessage() for r in engrava_records]}"
        )
        _assert_matches_allowed_content_free_shape(
            engrava_records,
            allowed_thought_ids=frozenset({thought_id}),
            allowed_exception_type_names=frozenset({type(original_exc).__name__}),
        )
        _assert_marker_absent_from_every_record(engrava_records, _MARKER)

    async def test_strict_mode(
        self,
        db: aiosqlite.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``require_embedding=True``: same content-free WARNING; a typed error is raised."""
        provider = _MarkerQuotingProvider()
        store = await _embedding_store(db, provider, require_embedding=True)
        thought_id = "t-single-strict"
        _widen_caplog(caplog)

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await store.create_thought(_thought(thought_id, _MARKER))

        assert len(provider.raised) == 1, "expected exactly one provider call to have raised"
        original_exc = provider.raised[0]
        assert exc_info.value.thought_id == thought_id
        assert exc_info.value.__cause__ is original_exc, (
            "EmbeddingGenerationError must chain from the provider's own exception"
        )
        assert str(exc_info.value) == str(
            EmbeddingGenerationError(thought_id, str(original_exc))
        ), "EmbeddingGenerationError must carry the provider's own message unchanged"

        engrava_records = _engrava_records(caplog.records)
        assert len(engrava_records) == 1, (
            f"expected exactly one engrava record (the warning), got {len(engrava_records)}: "
            f"{[r.getMessage() for r in engrava_records]}"
        )
        _assert_matches_allowed_content_free_shape(
            engrava_records,
            allowed_thought_ids=frozenset({thought_id}),
            allowed_exception_type_names=frozenset({type(original_exc).__name__}),
        )
        _assert_marker_absent_from_every_record(engrava_records, _MARKER)


# ---------------------------------------------------------------------------
# Batch path (bulk_store -> _batch_embed_thoughts)
# ---------------------------------------------------------------------------


class TestBatchPathNeverLogsProviderMessage:
    """``bulk_store``'s batch auto-embed failure never logs the provider's message."""

    async def test_default_mode(
        self,
        db: aiosqlite.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Default mode: the WARNING is content-free; the provider's own exception propagates."""
        provider = _MarkerQuotingProvider()
        store = await _embedding_store(db, provider)
        thought_id = "t-batch-default"
        _widen_caplog(caplog)

        with pytest.raises(ValueError) as exc_info:
            await store.bulk_store([_thought(thought_id, _MARKER)])

        assert len(provider.raised) == 1, "expected exactly one provider call to have raised"
        original_exc = provider.raised[0]
        assert exc_info.value is original_exc, (
            "the default path must propagate the provider's own exception by identity"
        )

        engrava_records = _engrava_records(caplog.records)
        assert len(engrava_records) == 1, (
            f"expected exactly one engrava record (the warning), got {len(engrava_records)}: "
            f"{[r.getMessage() for r in engrava_records]}"
        )
        _assert_matches_allowed_content_free_shape(
            engrava_records,
            allowed_thought_ids=frozenset({thought_id}),
            allowed_exception_type_names=frozenset({type(original_exc).__name__}),
        )
        _assert_marker_absent_from_every_record(engrava_records, _MARKER)

    async def test_strict_mode(
        self,
        db: aiosqlite.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``require_embedding=True``: same content-free WARNING; a typed error is raised."""
        provider = _MarkerQuotingProvider()
        store = await _embedding_store(db, provider, require_embedding=True)
        thought_id = "t-batch-strict"
        _widen_caplog(caplog)

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await store.bulk_store([_thought(thought_id, _MARKER)])

        assert len(provider.raised) == 1, "expected exactly one provider call to have raised"
        original_exc = provider.raised[0]
        assert exc_info.value.thought_id == thought_id
        assert exc_info.value.__cause__ is original_exc, (
            "EmbeddingGenerationError must chain from the provider's own exception"
        )
        assert str(exc_info.value) == str(
            EmbeddingGenerationError(thought_id, str(original_exc))
        ), "EmbeddingGenerationError must carry the provider's own message unchanged"

        engrava_records = _engrava_records(caplog.records)
        assert len(engrava_records) == 1, (
            f"expected exactly one engrava record (the warning), got {len(engrava_records)}: "
            f"{[r.getMessage() for r in engrava_records]}"
        )
        _assert_matches_allowed_content_free_shape(
            engrava_records,
            allowed_thought_ids=frozenset({thought_id}),
            allowed_exception_type_names=frozenset({type(original_exc).__name__}),
        )
        _assert_marker_absent_from_every_record(engrava_records, _MARKER)
