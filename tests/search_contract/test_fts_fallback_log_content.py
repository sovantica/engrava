"""Content-safety guard for the FTS5 ``MATCH``-failure log calls.

``SqliteEngravaCore.search_fts`` used to log the user's normalized query text
directly (via ``%r``) whenever the primary FTS5 ``MATCH`` failed and the
sanitizing bare-mode fallback ran, and it attached ``exc_info=True`` on top --
SQLite's own FTS5 syntax-error message can itself quote a fragment of the
offending expression. Under a wrapping MCP server, stderr lands in the host's
log files verbatim, so a failed search wrote whatever the caller searched for
straight into an operator's logs.

An intermediate fix replaced the query text with its length plus a truncated
SHA-256 digest for correlating repeats -- but a search query is often one or
two words, and an unsalted hash of low-entropy plaintext is a dictionary
lookup away from the original. The digest is gone; the log now carries only
the query length, the exception's type, and SQLite's own error name.

This suite checks two independent things about every record an engrava logger
emits while the fallback runs:

* **Shape (exact pin, not a shape match).** Every record from an engrava
  logger must carry ``record.msg`` identical to one of the two literal
  format strings the fix passes to ``logger.warning`` (copied from the
  source below, not reconstructed), and its rendering must satisfy
  ``record.getMessage() == record.msg % record.args`` -- confirming nothing
  else shapes the text a handler ultimately sees. The three substituted args
  are then drawn from closed sets rather than matched by shape: the query
  length must be a non-negative, non-``bool`` ``int`` equal to the real
  length of that branch's own normalized query (rejecting a content-derived
  stand-in such as ``hash(query)`` or ``sum(query.encode())`` -- neither one
  reliably fails a bare type/sign check alone); the exception type
  name must be one of ``sqlite3.Error``'s own class names (built from the
  module at import time, not hard-coded); and the SQLite error name must be
  one of ``sqlite3``'s own ``SQLITE_*`` constants (built from the module
  too). This is checked with ``caplog`` capturing *every* level (``NOTSET``
  on both the root logger and the ``engrava`` package logger, not just
  ``WARNING``), so an extra ``logger.debug(...)``/``logger.info(...)`` leak
  is seen too -- and because the args are closed-set membership tests rather
  than a "looks like an identifier" shape match, a wrong fix that glues
  extra content onto an otherwise-valid argument (e.g.
  ``f"{sqlite_errorname}_{digest}"``, which is still identifier-*shaped*)
  is rejected, not only a change to the literal message text.
* **Marker absence**, kept as a second, independent check: the literal
  marker text must not appear in any record's message, args, formatted
  exception, or other attributes, at any level -- catching a naive
  ``%r``/``str``-interpolation or ``extra=`` leak directly, redundantly with
  the shape check above.

The suite also forces the second, defense-in-depth branch -- the bare
fallback *also* failing, which the source marks unreachable for real input --
with a synthetic SQLite error constructed directly so its message quotes the
marker. The test does not depend on what a real SQLite error's message would
contain: it exercises this branch's own content discipline regardless. It
also proves the marker-absence assertion has real discriminating power: a
plausible wrong fix that drops the ``%r``-interpolated query from the message
but keeps ``exc_info=True`` is exactly the shape the formatted-exception
check exists to catch.
"""

from __future__ import annotations

import logging
import sqlite3
import traceback
from typing import TYPE_CHECKING

import pytest

from engrava.infrastructure.sqlite import engrava_core as core_mod

if TYPE_CHECKING:
    from engrava import SqliteEngravaCore

_LOGGER_NAME = core_mod.__name__

# A marker with no plausible overlap with FTS5 syntax, the fixture corpus
# vocabulary, or any log-format token engrava itself might emit.
_MARKER = "xqzk4f7pv9d3e1cnotamemoryword"

# The literal query this suite drives through the real fallback: a balanced
# quoted phrase wrapping the marker with a hazardous trailing ``?``.
_MARKER_QUERY = f'"{_MARKER}"?'

# ---------------------------------------------------------------------------
# Exact pin: the two literal format strings the fix passes to
# ``logger.warning``, copied verbatim from
# ``src/engrava/infrastructure/sqlite/engrava_core.py`` -- not reconstructed
# or matched by shape, so a wrong fix cannot satisfy this by producing
# something merely similar.
# ---------------------------------------------------------------------------

_ALLOWED_MESSAGE_FORMATS: tuple[str, ...] = (
    (
        "FTS MATCH failed for a query of length %d; retrying via "
        "sanitized bare-mode fallback [error type=%s, sqlite error=%s]"
    ),
    (
        "FTS bare-mode fallback also failed for a query of length %d; "
        "returning no FTS results [error type=%s, sqlite error=%s]"
    ),
)

# The one true length for each branch's log line, computed from the real
# normalizer functions (never restated by hand) so this tracks their actual
# behaviour: the primary failure logs the length of the normalized query
# (unchanged from ``_MARKER_QUERY`` for this expert-syntax shape), and the
# bare-fallback-also-failed branch logs the length of the *bare-normalized*
# query, which strips the surrounding quote and the trailing ``?``.
_EXPECTED_LENGTH_BY_FORMAT: dict[str, int] = {
    _ALLOWED_MESSAGE_FORMATS[0]: len(core_mod._normalize_fts_query(_MARKER_QUERY)),
    _ALLOWED_MESSAGE_FORMATS[1]: len(core_mod._normalize_fts_query_bare(_MARKER_QUERY)),
}


def _transitive_subclasses(cls: type) -> set[type]:
    """Return *cls* itself plus every subclass reachable through the hierarchy.

    Args:
        cls: The root class to walk from.

    Returns:
        *cls* together with every direct and indirect subclass.

    """
    result: set[type] = {cls}
    for subclass in cls.__subclasses__():
        result |= _transitive_subclasses(subclass)
    return result


# The closed set of names the exception-type arg may take: every class name
# in ``sqlite3``'s own ``Error`` hierarchy, built from the module so this
# tracks a future sqlite3 exception addition automatically instead of being
# retyped by hand and left to drift.
_ALLOWED_EXCEPTION_TYPE_NAMES: frozenset[str] = frozenset(
    cls.__name__ for cls in _transitive_subclasses(sqlite3.Error)
)

# The closed set of names the sqlite-error-name arg may take: every
# ``SQLITE_*`` integer result-code constant ``sqlite3`` exposes, built from
# the module for the same reason.
_ALLOWED_SQLITE_ERROR_NAMES: frozenset[str] = frozenset(
    name
    for name in dir(sqlite3)
    if name.startswith("SQLITE_") and type(getattr(sqlite3, name)) is int
)

# A record built the same way ``logging`` builds every real one, used only to
# read off which attribute names are part of the standard ``LogRecord`` shape
# -- so the allowlist below is derived from the real class, not retyped by
# hand and left to drift from it.
_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None))
)

# Attributes a *formatter* legitimately adds on top of the standard shape,
# even though nothing in this suite formats a record to text itself:
# - "message": set by ``logging.Formatter.format()`` as a side effect
#   (``record.message = record.getMessage()``); pytest's ``caplog`` handler
#   formats every record it captures (to build ``caplog.text``), so this is
#   present on every record reaching this assertion -- verified empirically.
# - "asctime": also set by ``Formatter.format()``, only when the active
#   format string references ``%(asctime)s``. This repo's caplog format does
#   not, so it never appears in practice, but it is allowed here too since it
#   is a formatter artifact, never a logging-call payload -- unlike an
#   ``extra={...}`` attribute, which is deliberately NOT allowlisted here.
_FORMATTER_SET_ATTRS: frozenset[str] = frozenset({"message", "asctime"})

_ALLOWED_RECORD_ATTRS: frozenset[str] = _STANDARD_LOG_RECORD_ATTRS | _FORMATTER_SET_ATTRS


def _engrava_records(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    """Return only the records emitted by an ``engrava``-namespaced logger.

    Args:
        records: All records ``caplog`` captured (may include third-party
            loggers once capturing is widened to every level).

    Returns:
        The subset whose logger name starts with ``"engrava"``.

    """
    return [record for record in records if record.name.startswith("engrava")]


def _assert_records_match_allowed_content_free_shape(
    records: list[logging.LogRecord],
) -> None:
    """Assert every record is exactly one of the two pinned, content-free shapes.

    Pins the record to the literal source, then closes off every remaining
    degree of freedom:

    * ``record.msg`` must be identical (``==``, not merely shaped like) one
      of :data:`_ALLOWED_MESSAGE_FORMATS`;
    * ``record.getMessage() == record.msg % record.args`` -- the rendered
      text is exactly what the pinned format string produces from the
      record's own args, nothing else shapes it;
    * ``record.args`` must be exactly a 3-tuple: the query length as a
      non-negative, non-``bool`` ``int``; then the exception type name as a
      member of :data:`_ALLOWED_EXCEPTION_TYPE_NAMES`; then the SQLite error
      name as a member of :data:`_ALLOWED_SQLITE_ERROR_NAMES`. Closed-set
      membership, not "looks like an identifier": a wrong fix that glues
      extra content onto an otherwise-valid argument (e.g.
      ``f"{sqlite_errorname}_{digest}"``) is not itself one of SQLite's own
      names, so it is rejected even though the composite string is still
      identifier-shaped;
    * ``record.exc_info`` must be ``None`` -- no attached traceback;
    * ``record.stack_info`` must be ``None`` -- no attached caller stack. A
      ``stack_info=True`` wrong fix puts the calling frames' source lines
      into the record, and a caller further up that passed a literal query
      would have it echoed there;
    * ``vars(record)`` must carry no attribute beyond a standard
      ``LogRecord``'s (see :data:`_ALLOWED_RECORD_ATTRS`) -- rejecting any
      ``extra=`` payload outright, whatever it is named or contains.

    Args:
        records: Records to check (typically pre-filtered to an engrava
            logger via :func:`_engrava_records`).

    Raises:
        AssertionError: If any record deviates from the shape above.

    """
    for record in records:
        assert record.msg in _ALLOWED_MESSAGE_FORMATS, (
            f"record format string is not one of the two the fix emits: {record.msg!r}"
        )

        args = record.args
        assert isinstance(args, tuple), f"expected a plain args tuple, got {args!r}"
        assert len(args) == 3, (
            f"expected exactly 3 args (length, exception type, sqlite error name), got {args!r}"
        )
        length, exc_type_name, sqlite_error_name = args

        rendered = record.msg % args
        assert record.getMessage() == rendered, (
            f"rendered message does not match the pinned format string and args: "
            f"{record.getMessage()!r} != {rendered!r}"
        )

        # ``type(x) is int`` (not ``isinstance``) deliberately excludes ``bool``,
        # a subclass of ``int`` in Python.
        assert type(length) is int, (
            f"expected the query length as a plain int (not bool), got "
            f"{length!r} ({type(length)!r})"
        )
        assert length >= 0, f"expected a non-negative query length, got {length!r}"

        assert exc_type_name in _ALLOWED_EXCEPTION_TYPE_NAMES, (
            f"{exc_type_name!r} is not one of sqlite3.Error's own class names: "
            f"{sorted(_ALLOWED_EXCEPTION_TYPE_NAMES)}"
        )

        assert sqlite_error_name in _ALLOWED_SQLITE_ERROR_NAMES, (
            f"{sqlite_error_name!r} is not one of sqlite3's own SQLITE_* names"
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


def _assert_length_arg_matches_normalized_query(
    records: list[logging.LogRecord],
) -> None:
    """Assert every record's length arg is the real length of its own branch's query.

    A length arg that merely passes ``type is int`` and ``>= 0`` (checked by
    :func:`_assert_records_match_allowed_content_free_shape`) still admits a
    content-derived fingerprint standing in for "a number that lets you
    correlate repeats": ``hash(query)`` can be negative, but
    ``sum(query.encode())`` never is, and neither one fails a type-and-sign
    check reliably. Comparing against the one true length for this suite's
    fixed marker query -- computed via the real normalizer functions in
    :data:`_EXPECTED_LENGTH_BY_FORMAT`, never restated by hand -- rejects any
    stand-in whose value differs from that length.

    Matched by the record's pinned format string, not by list position or
    order: the primary-failure and bare-fallback-also-failed records carry
    two different expected lengths, because the normalized query differs
    from its bare-normalized form.

    Args:
        records: Records to check (typically pre-filtered to an engrava
            logger, and already known to carry a pinned format string --
            see :func:`_assert_records_match_allowed_content_free_shape`).

    Raises:
        AssertionError: If any record's length arg is not exactly the real
            length for its branch.

    """
    for record in records:
        expected = _EXPECTED_LENGTH_BY_FORMAT.get(record.msg)
        assert expected is not None, f"no expected length registered for format {record.msg!r}"
        actual = record.args[0]
        assert actual == expected, (
            f"expected the query length {expected!r} for {record.msg!r}, got {actual!r}"
        )


# ---------------------------------------------------------------------------
# Marker absence: an independent, redundant check for literal content leaks.
# ---------------------------------------------------------------------------


def _raise_marker_quoting_sqlite_error() -> None:
    """Raise a synthetic ``OperationalError`` whose own message quotes the marker.

    Simulates the documented risk that SQLite's own FTS5 error text can name
    the offending fragment of a query, so the "keep ``exc_info``" wrong-fix
    scenario below has something concrete to leak.

    Raises:
        sqlite3.OperationalError: Always -- a synthetic, marker-quoting error.

    """
    message = f'fts5: syntax error near "{_MARKER}"'
    raise sqlite3.OperationalError(message)


def _assert_marker_absent_from_every_record(
    records: list[logging.LogRecord],
    marker: str,
) -> None:
    """Assert *marker* is absent from every record's message, args, traceback, and attributes.

    Checks four independent surfaces per record -- any one of which could
    carry content if it escaped the fix: the formatted message
    (``record.getMessage()``), the raw ``args`` stringified (in case content
    reached an ``args`` entry rather than the format string), the fully
    formatted exception -- ``record.exc_info`` rendered through
    :func:`traceback.format_exception`, which is what a handler actually
    writes to disk, not merely ``str(exception)`` -- and every other
    attribute on the record (``repr(vars(record))``), which is where a
    ``logging.warning(..., extra={...})`` payload lands: it never reaches
    ``getMessage()``, ``args``, or ``exc_info``, but a JSON log formatter
    can serialise it verbatim.

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
        if record.exc_info is not None:
            formatted = "".join(traceback.format_exception(*record.exc_info))
            assert marker not in formatted, f"marker leaked via formatted exception: {formatted!r}"
        record_dict_text = repr(vars(record))
        assert marker not in record_dict_text, (
            f"marker leaked via a record attribute (e.g. extra=): {record_dict_text!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrimaryFallbackNeverLogsQueryContent:
    """A real primary-``MATCH`` failure logs only the two known-safe, content-free shapes."""

    async def test_all_levels_match_allowlist_and_marker_absent(
        self,
        fts_store: SqliteEngravaCore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A genuine FTS5 syntax error triggers the fallback; nothing derived from it leaks.

        ``"<marker>"?`` is the same real, reachable shape the fallback fuzz
        suite pins as ``"forum"?``: a balanced quoted phrase (classifies
        expert) with a hazardous trailing ``?``. The primary ``MATCH``
        genuinely raises ``OperationalError`` against the real ``thought_fts``
        index and ``search_fts`` retries through the bare fallback.

        Capturing is widened to *every* level (not just ``WARNING``) on both
        the root logger and the ``engrava`` package logger, so a wrong fix
        that logs the query at ``INFO``/``DEBUG`` instead of ``WARNING``
        would still be seen here.
        """
        caplog.set_level(logging.NOTSET)
        caplog.set_level(logging.NOTSET, logger="engrava")

        results = await fts_store.search_fts(_MARKER_QUERY)

        assert isinstance(results, list)  # degrades, never raises
        # Filtered to engrava's own loggers: at NOTSET, aiosqlite's own DEBUG
        # tracing (its driver logs the full SQL and bound parameters, which
        # legitimately include the query) also lands in caplog.records.
        # Suppressing a third-party driver's own debug tracing is a host's
        # logging-configuration choice, not something engrava controls or
        # this fix is about -- only engrava's own log calls are in scope.
        engrava_records = _engrava_records(caplog.records)
        assert engrava_records, "expected the primary MATCH failure to log at least one record"
        _assert_records_match_allowed_content_free_shape(engrava_records)
        _assert_length_arg_matches_normalized_query(engrava_records)
        _assert_marker_absent_from_every_record(engrava_records, _MARKER)


class TestSecondaryFallbackNeverLogsQueryContent:
    """The bare-fallback-also-fails branch logs only the two known-safe, content-free shapes.

    This branch is marked unreachable for real input by ``search_fts``'s own
    source comment -- the bare path always sanitizes to a valid ``MATCH``. It
    is forced here as defense-in-depth verification only (not a reachability
    claim), with a synthetic SQLite error constructed directly so its message
    quotes the marker -- the test does not depend on what a real SQLite
    error's message happens to contain, only on this branch's own content
    discipline.
    """

    async def test_all_levels_match_allowlist_and_marker_absent(
        self,
        fts_store: SqliteEngravaCore,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force both the primary and the fallback ``MATCH`` to fail."""
        real_execute = fts_store._db.execute
        call_count = 0

        async def _fake_execute(sql: str, parameters: object = None) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                exc = sqlite3.OperationalError(f'fts5: syntax error near "{_MARKER}"')
                exc.sqlite_errorname = "SQLITE_ERROR"  # type: ignore[attr-defined]
                exc.sqlite_errorcode = 1  # type: ignore[attr-defined]
                raise exc
            return await real_execute(sql, parameters)

        monkeypatch.setattr(fts_store._db, "execute", _fake_execute)

        caplog.set_level(logging.NOTSET)
        caplog.set_level(logging.NOTSET, logger="engrava")

        results = await fts_store.search_fts(_MARKER_QUERY)

        assert results == []
        assert call_count == 2, "expected exactly the primary call and one fallback retry"
        # See the primary test above for why this is filtered to engrava's
        # own loggers rather than checked over every captured record.
        engrava_records = _engrava_records(caplog.records)
        assert len(engrava_records) >= 2, "expected a warning from both failed attempts"
        _assert_records_match_allowed_content_free_shape(engrava_records)
        _assert_length_arg_matches_normalized_query(engrava_records)
        _assert_marker_absent_from_every_record(engrava_records, _MARKER)


class TestAssertionDiscriminatesAPlausibleWrongFix:
    """The marker-absence assertion is not fooled by a fix that only edits the message."""

    def test_rejects_a_message_that_drops_the_query_but_keeps_exc_info(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A wrong fix that keeps ``exc_info=True`` is caught via the traceback.

        A plausible wrong fix removes the literal query from the format
        string but leaves ``exc_info=True`` attached -- and SQLite's own FTS5
        error message can quote a fragment of the offending expression right
        back. This reproduces exactly that shape (a content-free message plus
        a marker-carrying exception attached via ``exc_info``) and shows the
        marker-absence assertion still flags it via the formatted-exception
        check, not merely the message string.
        """
        logger = logging.getLogger(_LOGGER_NAME)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            try:
                _raise_marker_quoting_sqlite_error()
            except sqlite3.OperationalError:
                logger.warning(
                    "FTS MATCH failed; retrying via sanitized bare-mode fallback",
                    exc_info=True,
                )

        with pytest.raises(AssertionError, match="formatted exception"):
            _assert_marker_absent_from_every_record(caplog.records, _MARKER)
