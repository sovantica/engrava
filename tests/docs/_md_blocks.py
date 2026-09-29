"""Shared helpers for the documentation-example test suite.

The documentation tests treat the project's Markdown files (``README.md``
and everything under ``docs/``) as a source of executable truth. These
helpers locate the Markdown files and extract their fenced ``python``
code blocks so the individual test modules can compile, scan, or execute
them.

A fenced block follows CommonMark's fenced-code rule: an opener is a run of
three or more backticks or tildes (optionally followed by an info string,
e.g. ``python``; the info string of a backtick fence cannot itself contain a
backtick); the closer is a run of the same character at least as long as the
opener, with only spaces or tabs after it. Both the opener and the closer
may be indented by at most three spaces: a line indented four or more spaces
(or by a tab) is part of an indented code block, not a fence, so it neither
opens a block nor closes one. Only a space or a tab counts as whitespace for
any of this, and only ``\\n``, ``\\r\\n`` and ``\\r`` end a line: a no-break space,
a form feed, or a Unicode line separator after a closing fence keeps the line
from closing it, and stays in an info string. Leading Markdown blockquote
markers (``>``, each with an optional following space or tab) are stripped
**only when the fence's own opening line carries them** -- decided once per
block, from its opener, and then applied consistently to that block's closer
and body. A line of such a block with fewer markers ends the blockquote, so it
raises ``ValueError``. An ordinary (non-blockquoted) fence is parsed exactly as
written, so a literal ``>`` at the start of a body line (real content, not a
container marker) is preserved rather than silently discarded. Indented
blocks (up to three spaces) are supported; the captured body is dedented to
the fence's indentation. A fence that is opened but never closed before end of
file -- including one whose container prefix this scan cannot resolve
consistently between opener and closer -- raises ``ValueError`` rather than
silently dropping or misreading the block. So does a fence opened on the same
line as a list marker (``- ```python``): that opener sits inside a list item,
which this scan does not model, and reading its closer as an opener would
pair the later fences wrongly. Nor does it model HTML blocks or the indentation
of list items, so it raises ``ValueError`` for each of these: a fence-like line
while an HTML block may still be open; a
fence-like line that does not open a block (indented four or more columns, or
by a tab) in a document that has a list; a body line, in such a document, that
is less indented than its opener, that would close the fence if its
indentation were ignored, or that holds only whitespace beyond the opener's
indentation; a blockquoted opener, in such a document, whose markers are
indented; a tab inside the indentation of an indented fence's body; and a
body line of a blockquoted fence with a tab among its markers and leading
whitespace, because how much of that whitespace belongs to the block's text
depends on tab stops this scan does not carry into it. It raises
``ValueError`` as well for an info string that holds ``&`` or a backslash,
because CommonMark decodes character references and backslash escapes in an
info string and this scan keeps it as written, and for a NUL character in an
info string or a body line, because CommonMark reads it as U+FFFD. The
captured body is dedented by spaces only, so a no-break space or another
Unicode whitespace character at the start of a line stays in it.

Two entry points share the one fence scan: the ``extract_*`` functions return
the fenced blocks themselves, and :func:`lines_outside_fences` returns a
document's lines with every fenced block blanked out, for a check that must see
only the text a Markdown renderer would treat as prose (a table wrapped in a
fence is not a table).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

# tests/docs/_md_blocks.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"

# Markdown lets a fence use either backticks or tildes, three or more of
# them; a fence opened with one character type is closed only by the same
# type, by a run at least as long as the opener's.
_FENCE_CHARS = ("`", "~")
_MIN_FENCE_LENGTH = 3

# CommonMark lets an opening or closing fence be indented by at most three
# spaces; a line indented four or more is part of an indented code block.
_MAX_FENCE_INDENT = 3

# The only whitespace CommonMark recognises around a fence: a space or a tab.
# ``str.strip()`` with no argument also removes a no-break space, a form feed
# and every other Unicode whitespace character, so a line a renderer does not
# read as a fence would be read as one.
_SPACE_TAB = " \t"

# A tab advances to the next multiple of four columns.
_TAB_STOP = 4

# The only line endings CommonMark has. ``str.splitlines()`` also splits on
# vertical tab, form feed, the file/group/record separators, NEL and the
# Unicode line and paragraph separators, so it would find a fence boundary
# where a renderer sees one line.
_LINE_ENDING_RE = re.compile(r"\r\n|\r|\n")

_LIST_MARKER = r"(?:[-+*]|\d{1,9}[.)])[ \t]+"
_LIST_ITEM_FENCE_RE = re.compile(
    rf"^[ \t]*{_LIST_MARKER}(?:>[ \t]*|{_LIST_MARKER})*(?:`{{3,}}|~{{3,}})"
)

# The text of a line after its leading blockquote markers, list markers and whitespace.
_CONTAINER_PREFIX_RE = re.compile(r"(?:[ \t]*>|[ \t]*(?:[-+*]|\d{1,9}[.)])(?=[ \t]|$))*[ \t]*")
_LIST_LINE_RE = re.compile(r"(?:[ \t]*>)*[ \t]*(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)")

# A line that could start an HTML block. The kinds that end at a marker rather
# than at a blank line come first; anything else is read as ending at a blank
# line. ``(start, end, blank_too)``: ``blank_too`` keeps the block open after its
# end marker until a blank line, because older CommonMark specs may read a
# ``textarea`` block that way.
_HTML_START_RE = re.compile(r"<(?:[A-Za-z]|/[A-Za-z]|!|\?)")
_HTML_MARKED_KINDS = tuple(
    (re.compile(start, re.IGNORECASE), re.compile(end, re.IGNORECASE), blank_too)
    for start, end, blank_too in (
        (r"<(?:script|pre|style)(?:\s|>|$)", r"</(?:script|pre|style)>", False),
        (r"<textarea(?:\s|>|$)", r"</textarea>", True),
        (r"<!--", r"-->", False),
        (r"<\?", r"\?>", False),
        (r"<![A-Za-z]", r">", False),
        (r"<!\[CDATA\[", r"\]\]>", False),
    )
)


@dataclass(frozen=True)
class CodeBlock:
    """A fenced ``python`` code block extracted from a Markdown file.

    Attributes:
        path: Absolute path to the source Markdown file.
        rel: Path relative to the repository root (for messages).
        start_line: 1-based line number of the first body line.
        body: The dedented block body (without the fences).

    """

    path: Path
    rel: str
    start_line: int
    body: str

    @property
    def location(self) -> str:
        """Return a human-readable ``file:line`` locator."""
        return f"{self.rel}:{self.start_line}"


def markdown_files() -> list[Path]:
    """Return all documentation Markdown files in scope, sorted.

    Returns:
        ``README.md`` plus every ``*.md`` under ``docs/`` (recursive),
        in a stable sorted order so test parametrisation is
        deterministic.

    """
    files = [README, *sorted(DOCS_DIR.rglob("*.md"))]
    return [f for f in files if f.is_file()]


def extract_python_blocks(path: Path) -> list[CodeBlock]:
    """Extract every fenced ``python`` code block from one Markdown file.

    Args:
        path: The Markdown file to scan.

    Returns:
        A list of :class:`CodeBlock` in document order. Empty when the
        file contains no ``python`` fences.

    """
    return extract_fenced_blocks(path, "python")


def markdown_lines(text: str) -> list[str]:
    """Split ``text`` into lines the way CommonMark does.

    Only ``\\n``, ``\\r\\n`` and ``\\r`` end a line, and a line ending at the very end
    of ``text`` does not start another one (as with ``str.splitlines()``, whose
    other line boundaries this deliberately does not share).
    """
    lines = _LINE_ENDING_RE.split(text)
    if lines[-1] == "":
        lines.pop()
    return lines


def _strip_blockquote_prefix(line: str, depth: int = 1) -> str | None:
    """Strip ``depth`` leading Markdown blockquote markers (``>``) from a line.

    Each marker may be preceded by up to three spaces and is optionally
    followed by a single space, per CommonMark. A tab after a marker advances
    to the next tab stop, counted from the start of the original line, and the
    columns it spans past the one optional space stay as indentation.

    Returns:
        The line without those markers, or ``None`` when it carries fewer.
    """
    column = 0
    for _ in range(depth):
        lstripped = line.lstrip(" ")
        lead = len(line) - len(lstripped)
        if lead > _MAX_FENCE_INDENT or not lstripped.startswith(">"):
            return None
        after_marker = lstripped[1:]
        gap = len(after_marker) - len(after_marker.lstrip(_SPACE_TAB))
        marker_end = column + lead + 1
        cursor = marker_end
        expanded = ""
        for char in after_marker[:gap]:
            width = _TAB_STOP - cursor % _TAB_STOP if char == "\t" else 1
            expanded += " " * width
            cursor += width
        line = (expanded + after_marker[gap:]).removeprefix(" ")
        column = marker_end + (1 if expanded else 0)
    return line


def _fence_indent(line: str) -> int | None:
    """Return the number of spaces a fence line is indented by, or ``None`` if too deep.

    CommonMark allows at most three spaces before an opening or closing fence;
    four or more make the line part of an indented code block. A tab advances
    to the next multiple of four columns, so a leading tab is always too deep.
    """
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    if indent > _MAX_FENCE_INDENT or stripped.startswith("\t"):
        return None
    return indent


def _match_opening_fence(line: str) -> tuple[str, int, str] | None:
    """Return ``(fence_char, run_length, info_string)`` if the line opens a fence.

    The line must already have any container prefix (a blockquote marker)
    removed. It opens a fence only if it is indented by at most three spaces
    and continues with a run of at least three backticks or tildes; the info
    string of a backtick fence may not contain a backtick (that shape is inline
    code, not a fence).
    """
    if _fence_indent(line) is None:
        return None
    stripped_line = line.lstrip(" ")
    if not stripped_line or stripped_line[0] not in _FENCE_CHARS:
        return None
    char = stripped_line[0]
    length = len(stripped_line) - len(stripped_line.lstrip(char))
    if length < _MIN_FENCE_LENGTH:
        return None
    info = stripped_line[length:].strip(_SPACE_TAB)
    if char == "`" and "`" in info:
        return None
    return char, length, info


def _matches_closing_fence(line: str, fence_char: str, fence_length: int) -> bool:
    """Whether the line closes a fence opened with ``fence_char`` repeated ``fence_length`` times.

    Per CommonMark, the closer is indented by at most three spaces, then a run
    of the same character at least as long as the opener, followed by nothing
    but spaces or tabs. A line indented four or more spaces is content of the
    block, not its closer, and so is a line with any other character after the
    run, a no-break space included.
    """
    if _fence_indent(line) is None:
        return False
    stripped_line = line.lstrip(" ")
    if not stripped_line or stripped_line[0] != fence_char:
        return False
    run = len(stripped_line) - len(stripped_line.lstrip(fence_char))
    return run >= fence_length and stripped_line[run:].strip(_SPACE_TAB) == ""


class _FencedSpan(NamedTuple):
    """One fenced block found by :func:`_scan_fenced_blocks`."""

    info: str
    opener_index: int  # 0-based index of the opening fence line
    closer_index: int  # 0-based index of the closing fence line
    body: str

    @property
    def body_start_line(self) -> int:
        """1-based line number of the first body line."""
        return self.opener_index + 2


class _Opener(NamedTuple):
    """An opening fence line found by :func:`_find_opener`."""

    fence_char: str
    fence_length: int
    info: str
    line: str  # the line with its blockquote markers removed
    depth: int  # how many blockquote markers the line carried


def _find_opener(raw: str, index: int, source: str) -> _Opener | None:
    """Return the fence ``raw`` opens, or ``None`` if it opens none.

    Raises:
        ValueError: If the line opens a fence on the same line as a list marker.

    """
    depth = 0
    line = raw
    while True:
        fence = _match_opening_fence(line)
        if fence is not None:
            return _Opener(*fence, line, depth)
        if _LIST_ITEM_FENCE_RE.match(line):
            msg = (
                f"{source}: line {index + 1} opens a fence on a list-item line, which "
                f"this scan cannot follow; put the fence on its own line under the item"
            )
            raise ValueError(msg)
        depth += 1
        unquoted = _strip_blockquote_prefix(raw, depth)
        if unquoted is None:
            return None
        line = unquoted


def _markers_are_flush(raw: str, depth: int) -> bool:
    """Whether ``depth`` blockquote markers start ``raw``, each followed by at most one space."""
    rest = raw
    for _ in range(depth):
        if not rest.startswith(">"):
            return False
        rest = rest[1:].removeprefix(" ")
    return True


def _remainder(line: str) -> str:
    """Return ``line`` without its leading blockquote and list markers and whitespace."""
    match = _CONTAINER_PREFIX_RE.match(line)
    return line[match.end() :] if match else line


@dataclass
class _Unfollowed:
    """What the fence scan does not model, kept so it can refuse a line it cannot vouch for.

    The scan reads a document line by line and knows nothing of HTML blocks or of
    where a list item's content starts. Each line outside a fence is passed to
    :meth:`observe`, which remembers whether a list has appeared and which HTML
    blocks may still be open, and raises for a fence-like line that either could
    change the reading of.
    """

    list_seen: bool = False
    blank_ends_html: bool = False
    html_awaiting: list[tuple[re.Pattern[str], bool]] = field(default_factory=list)

    @property
    def html_open(self) -> bool:
        """Whether an HTML block that this line could be part of may be open."""
        return self.blank_ends_html or bool(self.html_awaiting)

    def observe(self, raw: str, index: int, source: str, *, opens_fence: bool) -> None:
        """Check one line outside a fenced block, then take it into account.

        Raises:
            ValueError: If the line looks like a fence but an HTML block may be
                open, or if it looks like a fence, does not open one and the
                document has a list.
        """
        rest = _remainder(raw)
        fence_like = _match_opening_fence(rest) is not None
        if fence_like and self.html_open:
            msg = (
                f"{source}: line {index + 1} looks like a fence while an HTML block may be "
                f"open, which this scan does not follow; end the HTML block first"
            )
            raise ValueError(msg)
        if fence_like and not opens_fence and self.list_seen:
            msg = (
                f"{source}: line {index + 1} looks like a fence but is indented four or more "
                f"columns (or by a tab) in a document that has a list, where a list item may "
                f"make it a fence; this scan cannot tell"
            )
            raise ValueError(msg)
        if _LIST_LINE_RE.match(raw):
            self.list_seen = True
        self._track_html(raw, rest)

    def check_opener(self, index: int, source: str, *, raw: str, depth: int) -> None:
        """Refuse a blockquoted opener whose markers are indented, in a document that has a list.

        Raises:
            ValueError: If the document has a list and a marker of the opener's blockquote
                is indented, so that a list item may hold the block.
        """
        if depth and self.list_seen and not _markers_are_flush(raw, depth):
            msg = (
                f"{source}: line {index + 1} opens a blockquoted fence whose blockquote markers "
                f"are indented, in a document that has a list; a list item may hold it and "
                f"this scan cannot tell where that item ends"
            )
            raise ValueError(msg)

    def _track_html(self, raw: str, rest: str) -> None:
        still_awaiting = []
        for end, blank_too in self.html_awaiting:
            if end.search(rest):
                self.blank_ends_html = self.blank_ends_html or blank_too
            else:
                still_awaiting.append((end, blank_too))
        self.html_awaiting = still_awaiting
        if raw.strip(_SPACE_TAB) == "":
            self.blank_ends_html = False
        if not _HTML_START_RE.match(rest):
            return
        for start, end, blank_too in _HTML_MARKED_KINDS:
            if start.match(rest):
                if end.search(rest):
                    self.blank_ends_html = self.blank_ends_html or blank_too
                else:
                    self.html_awaiting.append((end, blank_too))
                return
        self.blank_ends_html = True


def _check_info_string(index: int, source: str, info: str) -> None:
    """Refuse an info string that holds ``&``, a backslash or a NUL character.

    CommonMark decodes character references and backslash escapes in an info
    string (```` ```b&#97;sh ```` names the language ``bash``) and replaces NUL;
    the scan keeps the string as written.

    Raises:
        ValueError: If ``info`` holds ``&``, a backslash or a NUL character.
    """
    _check_no_nul(index, source, info)
    if "&" in info or "\\" in info:
        msg = (
            f"{source}: line {index + 1} opens a fenced block whose info string holds '&' or a "
            f"backslash, which CommonMark decodes and this scan does not"
        )
        raise ValueError(msg)


def _check_no_nul(index: int, source: str, raw: str) -> None:
    """Refuse a line that holds a NUL character.

    CommonMark replaces NUL with U+FFFD; the scan keeps the line as written.

    Raises:
        ValueError: If ``raw`` holds a NUL character.
    """
    if "\x00" in raw:
        msg = (
            f"{source}: line {index + 1} holds a NUL character in a fenced block, which "
            f"CommonMark reads as U+FFFD and this scan does not"
        )
        raise ValueError(msg)


def _body_line_may_move(effective: str, indent: int, fence_char: str, fence_length: int) -> bool:
    """Whether a list item's indentation could change how a fenced body line is read.

    A body line indented less than its opener may end the list item the block sits
    in, and a line that closes the fence only when its indentation is ignored may
    be at the right indentation for the item. A whitespace-only line keeps, in the
    scan, whatever is left after the opener's indentation, which a list item may
    drop.
    """
    if effective.strip(_SPACE_TAB) == "":
        return len(effective) > indent
    if len(effective) - len(effective.lstrip(" ")) < indent:
        return True
    return _matches_closing_fence(
        effective.lstrip(_SPACE_TAB), fence_char, fence_length
    ) and not _matches_closing_fence(effective, fence_char, fence_length)


def _scan_fenced_blocks(lines: list[str], source: str) -> list[_FencedSpan]:
    """Return every fenced block in ``lines``, in document order.

    The one scan every entry point below builds on -- implements
    CommonMark's fenced-code rule (see the module docstring) once, so a fix
    to fence detection (tildes, blockquotes, indentation, closer length) never
    needs to be made twice. ``source`` names the document in error messages.

    Whether a blockquote prefix is stripped is decided **once per block**,
    from its opener: a plain fence (the overwhelming common case) is parsed
    exactly as written, so a line starting with a literal ``>`` inside its
    body -- an ordinary character, not a container marker -- is never
    touched. Only a fence whose *own opening line* carries a ``>`` prefix has
    that prefix stripped, consistently, from its closer and its body too.

    Raises:
        ValueError: If a fence is opened but never closed before end of
            file -- Markdown lets a fence run to the end of the document,
            which would otherwise silently drop the block (and everything
            after it) instead of failing loudly. The same applies to a
            fence whose opener carries a container prefix (blockquote,
            optionally wrapping a list item) this scan cannot resolve
            consistently between the opener and the closer: it fails the
            same way, rather than silently mis-reading the block. So does a
            line with fewer blockquote markers than its block's opener, where
            the blockquote ends and CommonMark reads the line outside it. So does a
            fence opened on the same line as a list marker, which this scan
            cannot follow into the list item. It also raises for the info-string,
            NUL, HTML-block and list-item cases the module docstring lists.

    """
    results: list[_FencedSpan] = []
    unfollowed = _Unfollowed()
    in_block = False
    depth = 0
    fence_char = ""
    fence_length = 0
    indent = 0
    info = ""
    body_lines: list[str] = []
    opener_index = 0

    for index, raw in enumerate(lines):
        if not in_block:
            opener = _find_opener(raw, index, source)
            unfollowed.observe(raw, index, source, opens_fence=opener is not None)
            if opener is not None:
                unfollowed.check_opener(index, source, raw=raw, depth=opener.depth)
                _check_info_string(index, source, opener.info)
                fence_char, fence_length, info = opener.fence_char, opener.fence_length, opener.info
                in_block = True
                depth = opener.depth
                indent = len(opener.line) - len(opener.line.lstrip(" "))
                body_lines = []
                opener_index = index
            continue
        _check_no_nul(index, source, raw)
        effective = _strip_blockquote_prefix(raw, depth)
        if effective is None:
            msg = (
                f"{source}: line {index + 1} has fewer blockquote markers than the fenced "
                f"block opened at line {opener_index + 1} ({fence_char * fence_length}{info}), "
                f"so its blockquote ends before that block is closed"
            )
            raise ValueError(msg)
        closes = _matches_closing_fence(effective, fence_char, fence_length)
        if depth and not closes and "\t" in raw[: len(raw) - len(raw.lstrip(_SPACE_TAB + ">"))]:
            msg = (
                f"{source}: line {index + 1} has a tab among its blockquote markers and leading "
                f"whitespace inside the fenced block opened at line {opener_index + 1}, so how "
                f"much of it is the block's text depends on tab stops this scan does not carry"
            )
            raise ValueError(msg)
        lead = effective[: len(effective) - len(effective.lstrip(_SPACE_TAB))]
        if indent and "\t" in lead[:indent]:
            msg = (
                f"{source}: line {index + 1} has a tab inside the indentation of the fenced "
                f"block opened at line {opener_index + 1}, which this scan cannot dedent"
            )
            raise ValueError(msg)
        if unfollowed.list_seen and _body_line_may_move(
            effective, indent, fence_char, fence_length
        ):
            msg = (
                f"{source}: line {index + 1} is indented less than the fenced block opened at "
                f"line {opener_index + 1}, closes it only if its indentation is ignored, or "
                f"holds only whitespace beyond that indentation, in a document that has a "
                f"list; a list item may change how it is read"
            )
            raise ValueError(msg)
        if closes:
            body = "\n".join(_dedent(line, indent) for line in body_lines)
            results.append(_FencedSpan(info, opener_index, index, body))
            in_block = False
            continue
        body_lines.append(effective)

    if in_block:
        container = "blockquoted " if depth else ""
        msg = (
            f"{source}: a {container}fenced block opened at line {opener_index + 1} "
            f"({fence_char * fence_length}{info}) is never closed before end of file "
            f"(or its closer's container prefix does not match its opener's)"
        )
        raise ValueError(msg)

    return results


def _iter_fenced_blocks(path: Path) -> list[tuple[str, int, str]]:
    """Return ``(info_string, body_start_line, body)`` for every fenced block in a file.

    Raises:
        ValueError: If a fence is opened but never closed; see
            :func:`_scan_fenced_blocks`.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    lines = markdown_lines(path.read_text(encoding="utf-8"))
    return [
        (span.info, span.body_start_line, span.body) for span in _scan_fenced_blocks(lines, rel)
    ]


def lines_outside_fences(lines: list[str], source: str) -> list[str]:
    """Return ``lines`` with every fenced block blanked out, fences included.

    A line from a block's opening fence through its closing fence becomes an
    empty string, so line numbers and the position of everything else are
    unchanged while nothing inside a fence can be read as a heading, a table
    row, or any other Markdown structure. This is what a renderer does: text in
    a fence is code, whatever it looks like.

    Args:
        lines: The document's lines.
        source: The document's name, for the error message.

    Raises:
        ValueError: If a fence is opened but never closed; see
            :func:`_scan_fenced_blocks`.

    """
    blanked = list(lines)
    for span in _scan_fenced_blocks(lines, source):
        for index in range(span.opener_index, span.closer_index + 1):
            blanked[index] = ""
    return blanked


def extract_fenced_blocks(path: Path, language: str) -> list[CodeBlock]:
    """Extract every fenced block of one info-string language from a Markdown file.

    ``python`` blocks are the executable-example surface; other languages carry
    documented *output* (a ``text`` block showing what an example prints), which
    a test can compare against a real run so the page cannot promise a result it
    does not produce.

    Args:
        path: The Markdown file to scan.
        language: The fence info-string language, e.g. ``"python"`` or ``"text"``.
            Matched by *prefix*: ``"python"`` also matches a hypothetical
            ``python3`` fence. Use :func:`extract_exact_fenced_blocks` for an
            exact match.

    Returns:
        A list of :class:`CodeBlock` in document order. Empty when the file
        contains no fence of that language.

    Raises:
        ValueError: If a fence is opened but never closed before end of file.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    return [
        CodeBlock(path=path, rel=rel, start_line=start, body=body)
        for info, start, body in _iter_fenced_blocks(path)
        if info.startswith(language)
    ]


def all_python_blocks() -> list[CodeBlock]:
    """Return every ``python`` code block across all documentation files."""
    blocks: list[CodeBlock] = []
    for path in markdown_files():
        blocks.extend(extract_python_blocks(path))
    return blocks


def extract_all_fenced_blocks(path: Path) -> list[tuple[str, CodeBlock]]:
    """Extract every fenced code block in a Markdown file, with its exact info string.

    ``extract_fenced_blocks`` matches an info string by *prefix*: asking it for
    ``"python"`` also collects a hypothetical ```python3`` block, and asking it
    for the empty string collects **every** fenced block regardless of
    language (every fence "starts with" the empty string). That is fine for a
    single known language, but a census that must partition *every* fenced
    block -- including a bare fence carrying no language at all -- needs the
    info string matched exactly instead. This function is the one place that
    distinction is made.

    Args:
        path: The Markdown file to scan.

    Returns:
        ``(info_string, block)`` pairs in document order. ``info_string`` is
        ``""`` for a bare fence.

    Raises:
        ValueError: If a fence is opened but never closed before end of file.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    return [
        (info, CodeBlock(path=path, rel=rel, start_line=start, body=body))
        for info, start, body in _iter_fenced_blocks(path)
    ]


def extract_exact_fenced_blocks(path: Path, language: str) -> list[CodeBlock]:
    """Extract fenced blocks whose info string equals ``language`` exactly.

    Use this instead of :func:`extract_fenced_blocks` whenever the empty
    string must mean "a bare fence, no language" rather than "every fence" --
    see :func:`extract_all_fenced_blocks` for why the two disagree.

    Args:
        path: The Markdown file to scan.
        language: The exact info string to match, e.g. ``"bash"`` or ``""``
            for a bare fence.

    Returns:
        A list of :class:`CodeBlock` in document order.

    """
    return [block for info, block in extract_all_fenced_blocks(path) if info == language]


def all_blocks_by_exact_language() -> dict[str, list[CodeBlock]]:
    """Group every fenced block across all documentation files by exact info string.

    The keys are exact fence info strings (``""`` for a bare fence). This is
    the source of truth for a census over every language, not just ``python``.
    """
    grouped: dict[str, list[CodeBlock]] = defaultdict(list)
    for path in markdown_files():
        for info, block in extract_all_fenced_blocks(path):
            grouped[info].append(block)
    return grouped


# An exemption is bound to the block's text (line endings normalised) it was
# registered for. Sixteen hex characters of SHA-256 is plenty to notice an edit;
# this is tamper-evidence against a careless change, not a security boundary.
_DIGEST_LENGTH = 16


def block_digest(body: str) -> str:
    """Return the digest that binds an exemption to one block's text (line endings normalised).

    A registry that names an exempt block only by its file and an anchor phrase
    keeps exempting the block after it is edited: an unchecked command or an
    unvalidated key could be appended and every gate would stay green. Each
    exemption therefore also carries this digest of the block's text, applies
    only while the text still matches it, and is reported by
    :func:`exemption_digest_problems` once it does not.
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]


def exemption_digest_problems(
    registry: str,
    entries: Iterable[tuple[CodeBlock, str]],
) -> list[str]:
    """Describe every exemption whose block text no longer matches its registered digest.

    Args:
        registry: The name of the registry the digests live in, for the message.
        entries: ``(block, registered_digest)`` for every registered exemption.

    Returns:
        One message per stale entry, naming the block and giving the digest to
        register if the edit is deliberate. Empty when every entry still matches.

    """
    problems: list[str] = []
    for block, registered in entries:
        current = block_digest(block.body)
        if current != registered:
            problems.append(
                f"{block.location}: the block's text changed after it was exempted "
                f"(registered digest {registered!r}, current {current!r}). The exemption "
                f"no longer applies, so the block is checked again. If the edit is "
                f"deliberate and the block still cannot be checked, set its digest in "
                f"{registry} to {current!r}; otherwise fix the block."
            )
    return problems


class ExemptionReason(Enum):
    """Closed vocabulary for why a non-``python`` fenced block is not checked.

    A free-text reason field lets exemptions multiply invisibly -- the
    ``python`` census's 122 compile-only blocks hide behind 113 distinct
    free-text reasons, so nobody can count how many blocks are exempt for
    which cause, or notice one category quietly growing. Every non-``python``
    exemption in this suite cites one of these members instead, so a census
    test can print an exact tally per reason.

    Extend this enum deliberately when a genuinely new category of
    unaddressable block appears -- do not reach for a free-text string.
    """

    #: A ``bash`` line/block that does not invoke ``engrava`` at all (``pip
    #: install``, ``make``, ``sqlite3``, ``python -m ...``, ...).
    NOT_AN_ENGRAVA_INVOCATION = "not-an-engrava-invocation"
    #: A syntax-grammar template with placeholder tokens (``COMMAND``,
    #: ``<table>``, ``[WHERE <bool-expr>]``, ...), not a concrete example --
    #: it genuinely has the shape of an invocation/query, it just names no
    #: real command, option, or literal value.
    USAGE_GRAMMAR_PLACEHOLDER = "usage-grammar-placeholder"
    #: An architecture diagram, pipeline sketch, directory tree, or a
    #: transcript of real program output / an error message -- prose
    #: illustration, not a checkable claim about a command, flag, or key.
    DIAGRAM_OR_TRANSCRIPT = "diagram-or-transcript"
    #: A scoring formula or an algorithm sketch (loop/pseudocode) describing
    #: how a value is computed -- notation, not a diagram, transcript, or
    #: query.
    FORMULA_OR_PSEUDOCODE = "formula-or-pseudocode"
    #: A concrete, bare MindQL query example shown as language syntax, not
    #: wrapped in the ``engrava query "..."`` CLI invocation that would make
    #: it checkable.
    MINDQL_QUERY = "mindql-query"
    #: Raw SQL run directly against the SQLite file (a maintenance command or
    #: an escape-hatch query), not through any engrava-owned surface.
    RAW_SQL_ILLUSTRATION = "raw-sql-illustration"
    #: A block that shows the same top-level key twice, side by side, to
    #: illustrate alternate forms -- the real duplicate-key check correctly
    #: reports this as a duplicate key (it is one, by construction), so
    #: checking the whole block as one document can only ever report that,
    #: never validate either form on its own.
    DUPLICATE_KEY_ALTERNATE_FORMS = "duplicate-key-alternate-forms"


class CompileOnlyReason(Enum):
    """Closed vocabulary for why a compile-only ``python`` documentation block is not
    executed or behaviour-asserted.

    Not every member names something that genuinely *cannot* be executed --
    ``NO_ASSERTABLE_CLAIM``'s own definition says the block would run cleanly exactly
    as written, and ``HARNESS_SHAPE_MISMATCH`` names a limitation of the execute-layer
    harness, not a property of the example. "Not executed" is the accurate claim; "can
    never be executed" is not, and members should not be extended on the assumption
    that it is.

    Mirrors :class:`ExemptionReason` for the ``python`` fence's own, larger, previously
    unclassified tier: the 122 blocks in ``COMPILE_ONLY`` (see
    ``test_docs_examples_coverage.py``) used to carry 113 distinct free-text reasons --
    122 against 113 is nine duplicate uses, not a one-sentence-per-nine-blocks ratio --
    so nobody could count how many blocks were exempt for which cause, or notice one
    cause quietly growing. Every ``COMPILE_ONLY``
    entry now cites exactly one of these members in addition to its free-text note; the
    note may still say something the member cannot (which exact API, which specific
    test mirrors it) -- the member is what gets counted, the note is what gets read.

    Extend this enum deliberately when a genuinely new reason a block is not executed
    or behaviour-asserted appears -- never by grepping the old free-text sentence for
    a keyword.
    """

    #: A plain, default-configured store or connection is the block's only obstacle,
    #: and literal arguments (a string, an int, a default-valued keyword) cover
    #: everything else it needs. Concretely: building
    #: ``aiosqlite.connect(":memory:")`` with ``row_factory = aiosqlite.Row``, wrapping
    #: it in a bare ``SqliteEngravaCore(conn)``, and calling ``ensure_schema()`` -- with
    #: no extra constructor keyword, no wrapper class, and no other collaborator --
    #: reproduces everything the block's own undefined names need. A bare fragment
    #: that calls straight into ``store``/``conn`` (``await store.recall(...)``), or a
    #: helper function that takes one as its only out-of-the-ordinary parameter (its
    #: other parameters, if any, being plain literals such as a query string or a
    #: cycle number) and is never invoked in the block, both qualify. A block needing
    #: a *differently built* store (extra constructor keywords, a wrapper class) or
    #: any non-literal collaborator does not -- see ``REQUIRES_SPECIALLY_CONFIGURED_STORE``
    #: and ``UNDEFINED_DOMAIN_VALUE`` respectively.
    ASSUMES_STORE_OR_CONNECTION = "assumes-store-or-connection"

    #: The block needs more than a plain store/connection: an undefined value or
    #: callable standing in for domain-specific data a generic fixture cannot
    #: manufacture as a literal -- a specific id (``src_id``, ``some_thought_id``), a
    #: record or collection of records (``fact``, ``transient_thought``, the
    #: ``items``/``first``/``second``/``link`` arguments of a helper that builds or
    #: links them), a vector (``embedding``), a caller-supplied callback
    #: (``my_embed_fn``, ``my_llm``), a loop-control flag (``running``), a name only
    #: defined or imported in a sibling block on the same page (``observation``,
    #: ``RecencyBoostHooks``, ``RECENT_COMMAND``, an unimported ``ThoughtRecord``), or a
    #: second collaborator beyond the store itself (a caller-owned lock, an embedding
    #: provider instance).
    UNDEFINED_DOMAIN_VALUE = "undefined-domain-value"

    #: The block needs a store built differently from the plain default -- an extra
    #: constructor keyword that changes what the store can do (``journal_enabled=True``
    #: before ``store.journal`` is usable), or the store wrapped in a class that changes
    #: its behaviour (``ReadOnlyEngrava(store)`` before a write raises
    #: ``ReadOnlyViolationError``, which is the block's whole point). A plain
    #: ``SqliteEngravaCore(conn)`` with schema applied does not exhibit what the block
    #: demonstrates; a differently-configured one would.
    REQUIRES_SPECIALLY_CONFIGURED_STORE = "requires-specially-configured-store"

    #: The block's entire content is a definition -- a class, a ``Protocol``, a
    #: subclass, a bare method/function signature stub (body is ``...``), or a
    #: plugin/extension object -- never invoked or asserted within the block, so no
    #: collaborator would make it prove anything by itself.
    DEFINITION_ONLY = "definition-only"

    #: The block's own point is bound to a real on-disk path in a way a disposable,
    #: single-process fixture cannot cheaply substitute for -- not necessarily because
    #: the file must already exist (a fresh ``connect("engrava.db")`` call creates its
    #: own file just fine; that alone is not this reason). Concretely: an
    #: ``engrava.yaml`` config file ``from_config`` must read with real content, a data
    #: directory ``EngravaManager`` manages across several stores, on-disk migration
    #: SQL files referenced by a package-relative path, a corrupted on-disk journal to
    #: exercise a startup failure, or an example whose claim is that a named ``.db``
    #: file's data survives **across separate process invocations** -- a property one
    #: subprocess run cannot demonstrate no matter how cheaply it creates the file.
    REQUIRES_ON_DISK_ARTIFACT = "requires-on-disk-artifact"

    #: The block needs a live external system a test cannot cheaply fake: real model
    #: weights (``SentenceTransformerProvider``), a real network endpoint and API key
    #: (``OpenAICompatibleProvider``, ``HuggingFaceProvider``), or a running local
    #: service (``OllamaProvider``). An optional Python dependency that merely is not
    #: guaranteed installed (e.g. ``prometheus_client``) is not a live external system
    #: by itself -- no service, endpoint, credential, or network is involved, only a
    #: package that may be absent. A current entry citing this reason for that alone
    #: is a known open question, not a member of what this reason actually names.
    REQUIRES_LIVE_EXTERNAL_SERVICE = "requires-live-external-service"

    #: The block would run cleanly exactly as written -- no undefined collaborator, no
    #: disk, no network -- but asserts or prints nothing, so there is no claim for a
    #: test to check even though nothing prevents running it: a standalone capability
    #: probe against stdlib ``sqlite3``, a bare logging-configuration statement, or an
    #: object construction with no observable outcome.
    NO_ASSERTABLE_CLAIM = "no-assertable-claim"

    #: The block is self-contained and would behave correctly if run, but its shape
    #: does not fit the execute layer's required entrypoint convention (an
    #: ``asyncio.run(main())``-wrapped script) -- e.g. a synchronous class definition
    #: plus wiring that never awaits anything.
    HARNESS_SHAPE_MISMATCH = "harness-shape-mismatch"


def _dedent(line: str, indent: int) -> str:
    """Strip up to ``indent`` leading spaces from a captured body line.

    Only a space is removed: a no-break space or any other Unicode whitespace
    character at the start of the line belongs to the block's text.
    """
    leading_spaces = len(line) - len(line.lstrip(" "))
    return line[min(leading_spaces, indent) :]
