"""Couples exported exceptions to the ``## Exceptions`` table in api-reference.md.

**Scope boundary — read this before extending anything below.** This gate
checks exactly two things:

1. every exception exported via ``engrava.__all__`` is documented — some line
   in the Exceptions table's body has that name, and nothing else, as its
   first cell (optionally wrapped in one Markdown link — see ``_row_name``);
   and
2. no stale row remains — every such line names a currently exported
   exception, and every body line's first cell is a clean, single name cell
   (a row whose first cell carries extra prose, strikethrough, or a second
   token is caught by ``test_exceptions_table_rows_are_backtick_quoted``,
   which exists precisely so that shape cannot silently misdocument a
   *different* exception than the one the row is actually about — see
   ``_row_name`` and ``test_gate_rejects_a_deprecation_annotation_row``).

Discovery of "every exported exception" keys on real exception-ness
(``isinstance(obj, type) and issubclass(obj, BaseException)``), never on the
name ending in ``Error`` — ``test_gate_requires_a_row_for_an_exported_exception_not_named_error``
and ``test_gate_ignores_a_non_exception_class_named_like_an_error`` exist
specifically because a name-shape substitute for that check would leave every
other test in this file green (every currently exported exception happens to
end in ``Error``), so nothing else here would have proven the claim.

It does **not** validate Markdown table well-formedness. There is no fence
tracking, no escaped-pipe handling, and no general cell-splitting. A
malformed table renders visibly wrong in any Markdown viewer and is a problem
for the human reading the page, not for this gate. A few things below look
like structural parsing but are not: they are single, narrow predicates over
one line (or one cell) each, kept only because their absence is a real hole
this gate has had:

* **The header is identified by its first cell being exactly ``"Exception"``**
  (whitespace around it ignored), not by the whole line matching some fixed
  text. This is column-count agnostic on purpose — a maintainer adding an
  operational column (e.g. ``Retryable``) must not make this gate report the
  header missing (see ``test_gate_accepts_an_extra_header_column``). The
  search for that header is scoped to the ``## Exceptions`` section only
  (never the whole document) — a legitimate, unrelated table elsewhere with
  its own ``Exception``-first-cell header (e.g. an
  ``| Exception | Raised by |`` cross-reference table) must not read as a
  second candidate (see
  ``test_gate_accepts_a_legitimate_exception_table_elsewhere``).
* **The line directly under the header must itself look like a separator**
  (every cell, once trimmed, is one or more ``-`` optionally flanked by a
  single leading and/or trailing ``:``). Without this, a stale row can
  occupy the separator's position and be silently skipped, unread, by the
  body scan (see ``test_gate_rejects_a_stale_row_in_the_separators_place``).
* **A row's exception name is extracted from its first cell only if that
  cell, once trimmed and unwrapped from at most one surrounding Markdown
  link, is *nothing but* a single backtick-quoted token.** This is
  deliberately stricter than "the first backtick anywhere": a
  deprecation-annotation row like
  ``~~RemovedError~~ (use [`NewError`](new.md))`` is not a name cell — it
  has a strikethrough token, prose, and a link all in one cell — and must
  not be misread as documenting ``NewError`` while the fact that
  ``RemovedError`` no longer has its own row goes unnoticed. Link-unwrapping
  is the one normalisation kept deliberately, because a name like
  ``[`EngravaError`](errors.md#engravaerror)`` is still, unambiguously, a
  name and nothing else (see ``test_gate_accepts_a_linked_exception_name``).
  Do not extend this to emphasis, footnotes, or nested links — that is the
  parser creeping back in.

All three of the above operate on "the first cell", defined narrowly and only
for this purpose as the text between a line's first and second ``|`` (see
``_first_cell``) — this is not a general table-cell parser and is not used
for anything beyond identifying the header and extracting a row's name.

**Locating "the table body" uses one further structural fact, chosen because
it needs no parser: a blank line — or, equivalently, any line that does not
look like a ``|``-prefixed row — always ends a table.** The body is every
line from directly under the (now verified) separator up to the first line
that is not a ``|``-prefixed row. Every check above applies **only** to that
body: a second table, a note, a list, prose, or a stray ``|``-line sitting
anywhere else in the section is simply outside the body and is never looked
at (see ``test_gate_accepts_a_realistic_second_table_with_plain_text_rows``
and ``test_gate_accepts_prose_with_a_pipe_prefixed_line_in_the_section``).

Both the ``## Exceptions`` heading and the next ``## `` heading that
terminates the section tolerate incidental leading/trailing whitespace
(``.strip()``-equivalent matching on *both* boundaries, symmetrically) — this
is a content check, not a byte-for-byte text-identity check, and an indented
heading on either side is not worth a red build (see
``test_gate_accepts_an_indented_heading_and_indented_terminator``).

This narrow a scope is deliberate and cost something to arrive at. Review kept
finding a new way a *structural* model of "the table" could be
fooled — a decoy under a subheading, a commented-out table, a fenced table,
indentation edge cases, escaped pipes, a plausible second table with
plain-text rows, a stale row hiding in the separator's position, a linked
name, an extra column, a deprecation annotation, a same-shaped header
elsewhere, an asymmetric section boundary — because "correctly parse
Markdown" has no natural stopping point. The blank-line rule has one real,
accepted cost in exchange: a blank line accidentally inserted in the middle
of the real table truncates the body there, and every exception documented
below it reads as undocumented (see
``test_gate_rejects_a_blank_line_truncating_the_real_table``). That is
treated as a loud, correct failure, not a bug to route around — a human
fixes the blank line, the gate does not grow a merge-adjacent-fragments
heuristic to paper over it.

**The document must contain no HTML comment marker at all — a total ban,
not a scan for where a comment sits.** Three consecutive attempts at
scanning for comment placement (in the section only; before the heading too;
ordering-aware across a whole line) each closed one hiding case and left
another — a comment can hide a section from a rendered page regardless of
where it sits or how it is nested, and enumerating "where" is exactly the
kind of structural question this gate has already refused to answer for
tables. An HTML comment is invisible in the rendered document, and this gate
exists to ensure the exceptions documentation is *visible*, so this file
does not carry HTML comments — full stop. This trades away something real:
a balanced, purely editorial comment above the heading, which used to be
allowed, now fails too. That is deliberate, not an oversight; if a comment
is ever genuinely needed, the resulting red build is the conversation to
have about it, not a defect in this check.

There is deliberately no exemption or allow-list mechanism here. If a class
should not appear in the table, that is a decision for a human to make in
the table itself (or by not exporting the class) — not a list this test
reads around.
"""

from __future__ import annotations

import re
import sys
import warnings
from typing import TYPE_CHECKING

import pytest

import engrava
from tests.docs._md_blocks import REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path

API_REFERENCE = REPO_ROOT / "docs" / "api-reference.md"

# This module's own module object, so tests below can monkeypatch API_REFERENCE
# and _exported_exception_classes and have the *production* functions in this
# file (which look these names up as globals at call time) pick up the patch.
_THIS_MODULE = sys.modules[__name__]

_EXCEPTIONS_HEADING = "## Exceptions"
_H2_PREFIX = "## "
_HEADER_FIRST_CELL = "Exception"
_HTML_COMMENT_MARKER = "<!--"

# A GFM separator cell, once trimmed: one or more "-", optionally flanked by a
# single leading and/or trailing ":" (alignment markers ":---" / "---:" /
# ":---:" allowed). Deliberately does NOT accept a colon with no dash (": ")
# or a colon anywhere other than an edge ("--:--") -- those are not
# separators, whatever a looser character-class check might have let through.
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")

# A cell that is a single Markdown link wrapping its whole content, e.g.
# "[`EngravaError`](errors.md#engravaerror)". Captures the link's label.
# The destination group is a greedy ".*" anchored to the FINAL ")" (not
# "no ')' allowed"), so a destination that itself contains parentheses, e.g.
# "[`EngravaError`](errors_(legacy).md)", still matches instead of being
# silently rejected.
_WHOLE_CELL_LINK_RE = re.compile(r"^\[(.*)\]\(.*\)$")

# A cell that is nothing but a single backtick-quoted token, with nothing
# else before, after, or around it.
_EXACT_BACKTICK_CELL_RE = re.compile(r"^`([^`]+)`$")


def _display_path(path: Path) -> str:
    """Render ``path`` relative to the repo root when possible, else as-is.

    Falls back gracefully for a path outside the repo — e.g. the tmp_path
    decoy documents the gate-behaviour tests below construct — so an
    assertion message never crashes on the very inputs those tests exist to
    exercise.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _document_lines() -> list[str]:
    """Return every line of ``API_REFERENCE``, split on newlines."""
    return API_REFERENCE.read_text(encoding="utf-8").splitlines()


def _exceptions_heading_indices(lines: list[str]) -> list[int]:
    """Return the indices of every line matching the ``## Exceptions`` heading.

    Whitespace-tolerant (``.strip()``) on purpose: a heading with incidental
    leading or trailing whitespace is not worth a red build. This is a
    content check, not a byte-for-byte text-identity check.
    """
    return [index for index, line in enumerate(lines) if line.strip() == _EXCEPTIONS_HEADING]


def _exceptions_section_bounds(lines: list[str]) -> tuple[int, int]:
    """Return ``(start, end)`` bounding the lines of the ``## Exceptions`` section.

    ``start`` is the line right after the heading; ``end`` is the next H2
    heading's line, or ``len(lines)``. Fails loudly, with its own message, if
    the heading is missing or appears more than once — other pages link to
    it directly as an anchor, so it must name one unambiguous section. The
    terminating H2 match is whitespace-tolerant the same way the opening
    heading match is (both call ``.strip()`` before comparing) — an indented
    ``## Protocols`` still ends the section, symmetrically with an indented
    ``## Exceptions`` still starting one. A bare, title-less ``##`` also
    counts as a terminator, not only ``## `` followed by text.
    """
    indices = _exceptions_heading_indices(lines)
    if not indices:
        pytest.fail(f"{_display_path(API_REFERENCE)} has no {_EXCEPTIONS_HEADING!r} heading")
    if len(indices) > 1:
        pytest.fail(
            f"{_display_path(API_REFERENCE)} has {len(indices)} {_EXCEPTIONS_HEADING!r} "
            f"headings (at line(s) {[i + 1 for i in indices]}), expected exactly 1"
        )
    start = indices[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped == _H2_PREFIX.strip() or stripped.startswith(_H2_PREFIX):
            end = index
            break
    return start, end


def _assert_document_has_no_html_comment_marker(lines: list[str]) -> None:
    """Fail loudly if any line in the whole document contains ``<!--``.

    A total ban, not a scan for where a comment sits: an HTML comment is
    invisible in the rendered document, and this gate exists to ensure the
    exceptions documentation is *visible*, so this file carries no HTML
    comments at all. Three attempts at scanning for comment *placement*
    (in the section only; before the heading too; ordering-aware within a
    line) each closed one hiding case and left another open — a plain
    substring check over the whole document has no such gap, because it
    asks nothing about where or how a comment is written. The real
    ``docs/api-reference.md`` has zero ``<!--`` today, so this costs it
    nothing; if a comment is ever genuinely wanted, the resulting failure is
    the conversation to have, not a defect in this check.
    """
    offending = [line for line in lines if _HTML_COMMENT_MARKER in line]
    if offending:
        pytest.fail(
            f"{_display_path(API_REFERENCE)} contains an HTML comment marker ('<!--'): "
            f"{offending}. This file does not carry HTML comments — a comment is "
            "invisible in the rendered document, and this gate exists to ensure the "
            "exceptions documentation is visible. Remove it, or raise removing this "
            "convention as its own decision."
        )


def _is_pipe_line(line: str) -> bool:
    """Return whether ``line`` looks like a ``|``-prefixed table row.

    Whitespace-tolerant: incidental leading/trailing whitespace on an
    otherwise ``|``-prefixed line is not a reason to misjudge where a table
    starts, continues, or ends.
    """
    return line.strip().startswith("|")


def _first_cell(line: str) -> str | None:
    """Return the text between ``line``'s first and second ``|``, or ``None``.

    ``"| A | B |"`` returns ``" A "`` — everything up to, but not including,
    the *second* ``|`` — never the rest of the line. Deliberately narrow:
    this is not general cell-splitting (no escape handling, no column
    counting). It exists only to isolate the one cell the header's identity
    or a row's exception name is expected to live in — see
    ``_is_header_line`` and ``_row_name``. Returns ``None`` only when
    ``line`` (after trimming) does not start with ``|`` at all. A line with
    only one ``|`` in it at all, e.g. ``"| Exception"``, has no second ``|``
    to stop at, so this falls back to whatever follows that single ``|``
    instead of returning ``None`` — an unterminated cell is not specially
    rejected here; the caller's content comparison (exactly ``"Exception"``,
    or exactly one backtick token) is what actually rejects it, if it should
    be rejected.
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    parts = stripped.split("|", 2)
    return parts[1]


def _is_header_line(line: str) -> bool:
    """Return whether ``line``'s first cell is exactly ``"Exception"``.

    Column-count agnostic on purpose: a maintainer adding an operational
    column (e.g. a ``Retryable`` column) must not make this check report the
    header missing.
    """
    cell = _first_cell(line)
    return cell is not None and cell.strip() == _HEADER_FIRST_CELL


def _is_separator_line(line: str) -> bool:
    """Return whether ``line`` is a markdown header/body separator row.

    Every cell, once trimmed, must be one or more ``-`` optionally flanked
    by a single leading and/or trailing ``:`` (GFM alignment markers). A
    cell that is only whitespace/colons with no dash (``": "``), or has a
    colon anywhere other than an edge (``"--:--"``), is not a separator cell
    and makes the whole line not a separator. A single predicate over one
    line, not a table model — it exists only to stop a stale row from
    silently occupying the separator's position (immediately under the
    header) and being skipped, unread, by the body scan.
    """
    if not _is_pipe_line(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(cell) is not None for cell in cells)


def _row_name(line: str) -> str | None:
    """Return ``line``'s exception name if its first cell names exactly one.

    **Convention, stated and enforced, not an accident of a regex:** a name
    cell must be exactly one backtick-quoted exception name, optionally
    wrapped in a single Markdown link — nothing else. Bold, strikethrough,
    footnote markers, and any other formatting in that cell are not
    supported, on purpose: a name cell that could carry arbitrary formatting
    could never be told apart from a deprecation annotation like
    ``~~RemovedError~~ (use [`NewError`](new.md))`` — which contains a real
    backtick-quoted name but must not be read as documenting it, or the
    exception being deprecated silently loses its own row while the
    replacement reads as already documented.

    Concretely: the first cell (see ``_first_cell``) is trimmed and, if it is
    a single Markdown link wrapping its entire content (e.g.
    ``[`EngravaError`](errors.md#engravaerror)``, destination parentheses and
    all — see ``_WHOLE_CELL_LINK_RE``), unwrapped to the link's label. What
    remains must then be *nothing but* a single backtick-quoted token, or
    this returns ``None``.
    """
    cell = _first_cell(line)
    if cell is None:
        return None
    normalized = cell.strip()
    link_match = _WHOLE_CELL_LINK_RE.match(normalized)
    if link_match is not None:
        normalized = link_match.group(1).strip()
    match = _EXACT_BACKTICK_CELL_RE.match(normalized)
    return match.group(1) if match is not None else None


def _table_body() -> list[str]:
    """Return the Exceptions table's data rows.

    Confirms the whole document carries no HTML comment marker at all (see
    ``_assert_document_has_no_html_comment_marker``), then locates the one
    line, within the ``## Exceptions`` section only, whose first cell is
    exactly ``"Exception"`` (zero or more than one such line *in that
    section* is a document defect, reported by name — a same-shaped header
    belonging to some other, unrelated table elsewhere in the document is
    never even considered), confirms the line directly under it is a real
    separator row (not a stale row silently occupying that position), then
    takes every line from there up to the first line that does not look
    like a ``|``-prefixed row — a blank line ends a table, and so does
    anything else that is not a table row; both are treated identically.
    No column counting, no escape handling, no fence tracking, no notion of
    "a table" beyond these two checked lines.
    """
    lines = _document_lines()
    _assert_document_has_no_html_comment_marker(lines)
    start, end = _exceptions_section_bounds(lines)

    header_indices = [index for index in range(start, end) if _is_header_line(lines[index])]
    if not header_indices:
        pytest.fail(
            f"no line in the {_EXCEPTIONS_HEADING!r} section of "
            f"{_display_path(API_REFERENCE)} has its first cell equal to "
            f"{_HEADER_FIRST_CELL!r} (the Exceptions table header)"
        )
    if len(header_indices) > 1:
        pytest.fail(
            f"the {_EXCEPTIONS_HEADING!r} section of {_display_path(API_REFERENCE)} has "
            f"{len(header_indices)} lines whose first cell equals {_HEADER_FIRST_CELL!r} "
            f"(at line(s) {[i + 1 for i in header_indices]}), expected exactly 1"
        )
    # header_indices was built by scanning only range(start, end), so the one
    # match found is guaranteed to already be inside the section -- no
    # separate "is it in bounds" check is needed.
    header_index = header_indices[0]

    separator_index = header_index + 1
    separator_line = lines[separator_index] if separator_index < len(lines) else ""
    if not _is_separator_line(separator_line):
        pytest.fail(
            f"the line under the Exceptions header in {_display_path(API_REFERENCE)} "
            f"(line {separator_index + 1}) is not a header/body separator row: "
            f"{separator_line!r}. Add a '|---|---|...' separator directly under the header."
        )

    body: list[str] = []
    for line in lines[separator_index + 1 :]:
        if not _is_pipe_line(line):
            break
        body.append(line)
    return body


def _table_body_names() -> list[str]:
    """Return each table-body line's exception name, skipping lines with none.

    One name per line at most (see ``_row_name`` — a cell with more than one
    token, or extra prose around a token, yields no name at all, not
    multiple).
    """
    names: list[str] = []
    for line in _table_body():
        name = _row_name(line)
        if name is not None:
            names.append(name)
    return names


def _exported_exception_classes() -> dict[str, type[BaseException]]:
    """Return ``{name: class}`` for every ``engrava.__all__`` exception entry.

    Keyed on actual exception-ness (``isinstance(obj, type) and
    issubclass(obj, BaseException)``), not on the name ending in ``Error``.
    Some entries (e.g. the deprecated ``MindStore*`` aliases) resolve through
    the module's lazy ``__getattr__`` and emit a ``DeprecationWarning`` on
    access; that warning is expected here and not the thing under test.
    """
    exceptions: dict[str, type[BaseException]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name in engrava.__all__:
            obj = getattr(engrava, name)
            if isinstance(obj, type) and issubclass(obj, BaseException):
                exceptions[name] = obj
    return exceptions


def test_registry_is_nonempty() -> None:
    """Guard against the exported-exception scan silently finding nothing."""
    assert len(_exported_exception_classes()) >= 3, (
        "engrava.__all__ yielded fewer than 3 exception classes; the scan in "
        "this module (or the export list itself) likely broke."
    )


def test_every_exported_exception_has_a_table_row() -> None:
    """Every exception in ``engrava.__all__`` is named in the table body."""
    exported = _exported_exception_classes()
    names = set(_table_body_names())

    missing = sorted(set(exported) - names)
    assert not missing, (
        "these exceptions are exported via engrava.__all__ but are not named "
        f"in the Exceptions table body of {_display_path(API_REFERENCE)}: "
        f"{missing}. Add a row describing the condition and what a caller "
        "should do about it."
    )


def test_every_table_row_names_a_currently_exported_exception() -> None:
    """Every named body row names a real, currently exported exception.

    This is the check that is actually load-bearing for "a row survived its
    class being removed from ``engrava.__all__``": the name is extracted
    (a clean, single backtick-quoted cell) and simply is not in the exported
    set anymore.
    """
    exported = _exported_exception_classes()
    names = [name for line in _table_body() if (name := _row_name(line)) is not None]

    stale = sorted(set(names) - set(exported))
    assert not stale, (
        f"the Exceptions table body of {_display_path(API_REFERENCE)} has "
        f"row(s) that no longer name an exported exception: {stale}. Remove "
        "the row, or restore the export if that was accidental."
    )


def test_exceptions_table_rows_are_unique() -> None:
    """The Exceptions table body names each exception at most once."""
    names = _table_body_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, (
        f"the Exceptions table body of {_display_path(API_REFERENCE)} lists "
        f"these exception(s) more than once: {duplicates}. Remove the duplicate row."
    )


def test_exceptions_table_rows_are_backtick_quoted() -> None:
    """Every body line's first cell is a clean, single backtick-quoted name.

    This check's remaining value is narrower than it looks, now that the
    forward check above exists: a body row for a *currently exported*
    exception whose cell stops being a clean name (backticks lost entirely,
    or extra prose/strikethrough/a second token added around a real name) is
    already caught there too — the name is no longer extracted, so that
    exception reads as undocumented, and the load-bearing check for *that*
    consequence is ``test_every_exported_exception_has_a_table_row``, not
    this one. What this check adds on top is the doubly-degenerate case: a
    row that is *both* no longer exported *and* has an unclean cell, which
    the reverse check above cannot see at all (it extracts names via the
    same rule, so an unclean cell is invisible to it too). That is a narrow,
    rare intersection, but the check is harmless and correctly scoped to the
    body now, so it stays.
    """
    malformed = [line for line in _table_body() if _row_name(line) is None]
    assert not malformed, (
        f"the Exceptions table body of {_display_path(API_REFERENCE)} has row(s) whose "
        f"first cell is not a name cell: {malformed}. A name cell must be exactly one "
        f"backtick-quoted exception name, optionally wrapped in a single Markdown link "
        f"— bold, strikethrough, and footnote markers are not supported in that cell."
    )


def test_real_documentation_table_body_is_nonempty() -> None:
    """Sanity check: the real document's table-body extraction is not vacuous.

    This does not by itself exercise every check above — the four tests
    above already run against the real, un-mocked ``API_REFERENCE`` whenever
    the suite runs normally, which is what actually proves the real document
    satisfies them. This test only guards against the narrower failure mode
    of ``_table_body_names()`` silently returning an empty list (e.g. from a
    bound or separator regression that still technically returns "no rows"
    instead of failing loudly).
    """
    names = _table_body_names()
    assert names, "the real Exceptions table body produced no exception names at all"


# ---------------------------------------------------------------------------
# Gate-behaviour pins: build every counterexample raised in review as a real
# decoy document (via monkeypatching API_REFERENCE, and where needed
# _exported_exception_classes) and assert the real production test functions
# above actually go red — or, for the false-red guards, stay green — on it.
# Each false-red guard below is built from a REALISTIC shape (plain-text
# rows, prose, no backticks anywhere) rather than one that happens to satisfy
# the very rules it is supposed to test around — a control built from the
# happy path proves nothing.
# ---------------------------------------------------------------------------


def _write_decoy(tmp_path: Path, body: str) -> Path:
    decoy = tmp_path / "api-reference.md"
    decoy.write_text(body, encoding="utf-8")
    return decoy


def test_gate_accepts_a_realistic_second_table_with_plain_text_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legitimate second table with plain-text (non-backtick, non-exception)
    rows, inside the same section, must be accepted. Unlike a control whose
    second table happens to use backtick-quoted, currently-exported names in
    its own first column (which would pass every check by pure accident),
    this is the shape a real second table actually has, and it is exactly
    the decoy that found this gate's earlier false red.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "### Retry policy\n"
        "\n"
        "| Situation | Suggested retry |\n"
        "|-----------|------------------|\n"
        "| contention | retry once |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: the retry-policy table is entirely outside the body
    # (the body ends at the blank line right after EngravaError's row).
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_accepts_prose_with_a_pipe_prefixed_line_in_the_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray ``|``-prefixed line of prose elsewhere in the section (not
    attached to the real table) must not be scanned at all.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "| this reads like a table row but is just a stray line of prose |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: the stray line is past the blank line that ends the body.
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_accepts_an_extra_header_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A maintainer adding an operational column (e.g. ``Retryable``) must not
    make the header read as missing. Header identification is column-count
    agnostic: only the first cell being ``"Exception"`` matters.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Retryable | Description |\n"
        "|-----------|------|-----------|-------------|\n"
        "| `EngravaError` | `Exception` | No | Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: the extra "Retryable" column does not change the first cell.
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_accepts_a_linked_exception_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name formatted as a Markdown link around backticks, e.g.
    ``[`EngravaError`](errors.md#engravaerror)``, is still — once the one
    surrounding link is unwrapped — nothing but a single backtick-quoted
    name, and must still count.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| [`EngravaError`](errors.md#engravaerror) | `Exception` | "
        "Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: one layer of link-wrapping is unwrapped before checking.
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_accepts_a_linked_exception_name_with_a_parenthesised_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A link whose destination itself contains parentheses, e.g.
    ``[`EngravaError`](errors_(legacy).md)``, must still be unwrapped
    correctly — the destination is not required to be parenthesis-free.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| [`EngravaError`](errors_(legacy).md) | `Exception` | "
        "Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: the destination's own parentheses do not break unwrapping.
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_accepts_a_legitimate_exception_table_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legitimate, unrelated table elsewhere in the document that happens to
    also have ``"Exception"`` as its first-cell header (e.g. a cross-reference
    table like ``| Exception | Raised by |``) must not read as a second
    header candidate. The header search is scoped to the ``## Exceptions``
    section only.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "## Somewhere else\n"
        "\n"
        "| Exception | Raised by |\n"
        "|-----------|-----------|\n"
        "| `EngravaError` | `create_edge` |\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: the cross-reference table's header is outside the
    # '## Exceptions' section and is never even considered.
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_accepts_an_indented_heading_and_indented_terminator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An indented ``## Exceptions`` heading and an equally-indented
    ``## Protocols`` terminator must both be recognised — symmetrically.
    Unrelated content past the (correctly recognised) terminator must not be
    mistakenly attributed to the section.
    """
    decoy = _write_decoy(
        tmp_path,
        "  ## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "  ## Protocols\n"
        "\n"
        "An ordinary paragraph in an unrelated section, not this gate's business.\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    # Must not raise: the indented terminator ends the section before the
    # unrelated paragraph, so nothing there is attributed to '## Exceptions'.
    test_every_exported_exception_has_a_table_row()
    test_every_table_row_names_a_currently_exported_exception()
    test_exceptions_table_rows_are_backtick_quoted()


def test_gate_rejects_an_exported_exception_with_no_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported exception with no row in the table body must fail."""
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception, "UndocumentedError": Exception},
    )

    with pytest.raises(AssertionError, match="UndocumentedError"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_a_stale_body_row_naming_a_non_exported_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body row naming a class that is no longer exported must fail."""
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "| `RemovedError` | `Exception` | No longer exported, row left behind |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    with pytest.raises(AssertionError, match="RemovedError"):
        test_every_table_row_names_a_currently_exported_exception()


def test_gate_rejects_a_duplicate_body_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same exception named twice in the table body must fail, naming it."""
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "| `EngravaError` | `Exception` | Listed twice by mistake |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    with pytest.raises(AssertionError, match="EngravaError"):
        test_exceptions_table_rows_are_unique()


def test_gate_rejects_a_body_row_that_lost_its_backticks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body row whose first cell is not backtick-quoted at all must fail."""
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "| OldError | `Exception` | stale row, lost its backticks in an edit |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)

    with pytest.raises(AssertionError, match="OldError"):
        test_exceptions_table_rows_are_backtick_quoted()


def test_gate_rejects_a_first_cell_with_trailing_prose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first cell that names a real, backtick-quoted exception but also
    carries extra prose (e.g. an inline deprecation note) is not a clean
    name cell and must fail — the name is not extracted from it at all.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` (deprecated) | `Exception` | Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)

    with pytest.raises(AssertionError, match=re.escape("`EngravaError` (deprecated)")):
        test_exceptions_table_rows_are_backtick_quoted()


def test_gate_rejects_a_deprecation_annotation_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row whose first cell mixes a strikethrough old name, prose, and a
    link to the replacement (e.g. ``~~RemovedError~~ (use [`NewError`](x))``)
    must not be read as documenting the replacement: the replacement then has
    no row of its own and must be reported missing. This is the exact defect
    a prior, looser rule ("the first backtick anywhere in the cell") allowed
    (verified separately, outside this suite, to actually pass under that
    prior rule — see the commit message for that discrimination check).
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| ~~RemovedError~~ (use [`NewError`](new.md)) | `Exception` | deprecated |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"NewError": Exception},
    )

    with pytest.raises(AssertionError, match="NewError"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_a_stale_row_in_the_separators_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale row sitting where the separator should be, with the real
    separator missing entirely, must fail by naming that line — not be
    silently skipped as if it were the separator.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "| `RemovedError` | `Exception` | stale |\n"
        "| `EngravaError` | `Exception` | current |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    with pytest.raises(pytest.fail.Exception, match="is not a header/body separator row"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_a_separator_line_with_a_colon_but_no_dash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A separator-position line where a cell has a colon with no dash, or a
    colon anywhere other than an edge, must be rejected as a separator.

    ``| : | --:-- | - |`` is exactly the shape a looser character-class
    check (``^[\s:-]+$`` per cell, tried and reverted) would have wrongly
    accepted: every character in it is one of "-", ":", or whitespace, so
    that check alone cannot tell it apart from a real separator. The current
    per-cell rule (``^:?-+:?$``: one or more "-", optionally flanked by a
    single leading and/or trailing ":") rejects both ":" (no dash at all)
    and "--:--" (colon not at an edge), so this line must never be silently
    treated as the separator position.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "| : | --:-- | - |\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)

    with pytest.raises(pytest.fail.Exception, match="is not a header/body separator row"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_a_duplicated_header_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two lines whose first cell is ``"Exception"``, both inside the
    ``## Exceptions`` section — even a second one only present inside a
    fenced example — leave no single, unambiguous table.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "A fenced example that happens to contain a header-shaped line:\n"
        "\n"
        "```markdown\n"
        "| Exception | Base | Description |\n"
        "```\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)

    with pytest.raises(pytest.fail.Exception, match=r"has 2 lines whose first cell equals"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_an_absent_header_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No line in the ``## Exceptions`` section has ``"Exception"`` as its
    first cell at all.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Situation | Base |\n"
        "|-----------|------|\n"
        "| `EngravaError` | `Exception` |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)

    with pytest.raises(
        pytest.fail.Exception,
        match=r"no line in the '## Exceptions' section .* has its first cell equal to",
    ):
        test_every_exported_exception_has_a_table_row()


def test_gate_ignores_a_header_line_outside_the_exceptions_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A header-shaped line existing only under the wrong heading is not
    found at all — the search never looks outside the ``## Exceptions``
    section — so this now shares the "no header in the section" failure
    with a truly absent header, which is the correct, simpler outcome now
    that the search is scoped rather than global.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "Nothing here.\n"
        "\n"
        "## Somewhere else\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)

    with pytest.raises(
        pytest.fail.Exception,
        match=r"no line in the '## Exceptions' section .* has its first cell equal to",
    ):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_a_blank_line_truncating_the_real_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank line accidentally inserted in the middle of the real table
    truncates the body there — the honest, accepted consequence of the
    blank-line rule. Rows below it become invisible, so their exceptions
    read as undocumented: a loud failure, not a silent partial read.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "| `ThoughtNotFoundError` | `EngravaError` | Thought ID not found |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception, "ThoughtNotFoundError": Exception},
    )

    with pytest.raises(AssertionError, match="ThoughtNotFoundError"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_any_html_comment_marker_anywhere_in_the_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTML comment anywhere in the document -- even far from the
    Exceptions section entirely -- must fail. This file carries no HTML
    comments at all, regardless of where one appears or how it is nested;
    there is no "where" question left to ask.
    """
    decoy = _write_decoy(
        tmp_path,
        "## Somewhere else entirely\n"
        "\n"
        "<!-- an ordinary editorial note, nowhere near Exceptions -->\n"
        "\n"
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    with pytest.raises(pytest.fail.Exception, match="contains an HTML comment marker"):
        test_every_exported_exception_has_a_table_row()


def test_gate_rejects_a_comment_that_closes_and_reopens_on_one_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single line that closes one comment and opens a second
    (``<!-- old note --> <!-- temporarily hide section``), immediately above
    the heading, used to defeat an ordering-aware scanner that tracked only
    one open/close flag per line: it saw one opener and one closer and
    called the line clear, even though a new, still-open comment starts on
    it and is not closed until much later. The whole-document ban needs
    none of that reasoning -- the line contains ``<!--``, which is already
    disallowed, independent of how many comments are on it or in what order
    they open and close.
    """
    decoy = _write_decoy(
        tmp_path,
        "<!-- old note --> <!-- temporarily hide section\n"
        "## Exceptions\n"
        "\n"
        "| Exception | Base | Description |\n"
        "|-----------|------|-------------|\n"
        "| `EngravaError` | `Exception` | Base for all engrava errors |\n"
        "-->\n"
        "## Protocols\n",
    )
    monkeypatch.setattr(_THIS_MODULE, "API_REFERENCE", decoy)
    monkeypatch.setattr(
        _THIS_MODULE,
        "_exported_exception_classes",
        lambda: {"EngravaError": Exception},
    )

    with pytest.raises(pytest.fail.Exception, match="contains an HTML comment marker"):
        test_every_exported_exception_has_a_table_row()


# ---------------------------------------------------------------------------
# Discovery-logic pins: prove _exported_exception_classes keys on real
# exception-ness (issubclass(BaseException)), not on the name ending in
# "Error". Unlike the decoys above, these monkeypatch the real `engrava`
# module's attributes and __all__ (not _exported_exception_classes itself),
# so the function's own real logic runs and is what gets proven.
# ---------------------------------------------------------------------------


class _Failure(Exception):  # noqa: N818 -- the point is that it does NOT end in "Error"
    """A real exception whose name does not end in "Error", used only below."""


class _LooksLikeError:
    """A non-exception class whose name ends in "Error", used only below."""


def test_gate_requires_a_row_for_an_exported_exception_not_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported exception under a name not ending in "Error" must still be
    required to have a row. If discovery were keyed on
    ``name.endswith("Error")`` instead of ``issubclass(BaseException)``, this
    class would be silently skipped and could never be reported missing.
    """
    monkeypatch.setattr(engrava, "__all__", [*engrava.__all__, "Failure"])
    monkeypatch.setattr(engrava, "Failure", _Failure, raising=False)

    exported = _exported_exception_classes()
    assert "Failure" in exported, (
        "the real exported-exception scan did not pick up an exported "
        "exception whose name does not end in 'Error'; discovery must key "
        "on issubclass(BaseException), not on the name"
    )


def test_gate_ignores_a_non_exception_class_named_like_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-exception class whose name ends in "Error" must be ignored
    entirely by exported-exception discovery.
    """
    monkeypatch.setattr(engrava, "__all__", [*engrava.__all__, "LooksLikeError"])
    monkeypatch.setattr(engrava, "LooksLikeError", _LooksLikeError, raising=False)

    exported = _exported_exception_classes()
    assert "LooksLikeError" not in exported, (
        "the real exported-exception scan picked up a non-exception class "
        "purely because its name ends in 'Error'; discovery must key on "
        "issubclass(BaseException), not on the name"
    )
