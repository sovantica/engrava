"""Unit tests for the fenced-block extractor every documentation gate shares.

The example gates (shell, config, python, output) and the exceptions-table gate
all decide what is a code block, and what is not, with
``tests.docs._md_blocks``. A gate is only as strict as that decision: an
extractor that lets four spaces open a fence, or lets a four-space line close
one, reads a different document from the one a reader sees, and every gate
built on it inherits the error. These tests pin the CommonMark rules the
extractor follows (an opener or closer indented at most three spaces; a closer
of the same character at least as long as the opener, followed by nothing but
spaces or tabs; a backtick fence's info string free of backticks; a line ended
only by ``\\n``, ``\\r\\n`` or ``\\r``), through the entry points the gates use, on
small documents written to ``tmp_path``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs import _md_blocks
from tests.docs._md_blocks import (
    CodeBlock,
    block_digest,
    exemption_digest_problems,
    extract_all_fenced_blocks,
    lines_outside_fences,
    markdown_lines,
)


def _blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> list[tuple[str, CodeBlock]]:
    """Extract every fenced block from ``text``, written to a page under ``tmp_path``."""
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    page = tmp_path / "page.md"
    page.write_text(text, encoding="utf-8")
    return extract_all_fenced_blocks(page)


# ---------------------------------------------------------------------------
# Opener: indented at most three spaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("indent", [0, 1, 2, 3])
def test_an_opener_indented_up_to_three_spaces_opens_a_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    indent: int,
) -> None:
    pad = " " * indent
    found = _blocks(monkeypatch, tmp_path, f"{pad}```python\n{pad}x = 1\n{pad}```\n")

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


@pytest.mark.parametrize("indent", ["    ", "     ", "        ", "\t", " \t"])
def test_an_opener_indented_four_spaces_or_a_tab_does_not_open_a_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    indent: str,
) -> None:
    """Four spaces (or a tab) make an indented code block, so this is no fence.

    The closing line is indented the same way, so the document has no fenced
    block at all, and in particular does not fail as "opened and never closed".
    """
    found = _blocks(monkeypatch, tmp_path, f"{indent}```python\n{indent}x = 1\n{indent}```\n")

    assert found == []


def test_a_deeply_indented_fence_does_not_hide_the_block_after_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A four-space "opener" must not swallow the real block that follows it."""
    text = "    ```bash\n\n```python\nx = 1\n```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


# ---------------------------------------------------------------------------
# Closer: indented at most three spaces, same character, at least as long
# ---------------------------------------------------------------------------


def test_a_four_space_line_does_not_close_a_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The "closing" line is indented four spaces, so it is body text; the block goes on."""
    text = "```python\na = 1\n    ```\nb = 2\n```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == [
        ("python", "a = 1\n    ```\nb = 2"),
    ]


def test_a_tab_indented_line_does_not_close_a_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text = "```python\na = 1\n\t```\nb = 2\n```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == ["a = 1\n\t```\nb = 2"]


def test_a_four_space_closer_alone_leaves_the_block_unclosed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no other closer the block never ends, and the extractor says so."""
    with pytest.raises(ValueError, match="never closed"):
        _blocks(monkeypatch, tmp_path, "```python\na = 1\n    ```\n")


@pytest.mark.parametrize("indent", [1, 2, 3])
def test_a_closer_indented_up_to_three_spaces_closes_the_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    indent: int,
) -> None:
    pad = " " * indent
    text = f"```python\na = 1\n{pad}```\nafter = 1\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == ["a = 1"]


# ---------------------------------------------------------------------------
# Backtick and tilde fences, info strings, closer length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opener", "closer"),
    [
        ("```python", "```"),
        ("~~~python", "~~~"),
        ("````python", "````"),
        ("~~~~python", "~~~~"),
        ("````python", "`````"),
    ],
    ids=["backtick", "tilde", "four-backticks", "four-tildes", "longer-closer"],
)
def test_backtick_and_tilde_fences_open_and_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
    closer: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, f"{opener}\nx = 1\n{closer}\n")

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


def test_a_tilde_fence_is_not_closed_by_backticks_and_the_reverse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text = "~~~text\n```\nstill body\n```\n~~~\n```text\n~~~\nalso body\n~~~\n```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [
        "```\nstill body\n```",
        "~~~\nalso body\n~~~",
    ]


def test_a_closer_shorter_than_the_opener_does_not_close_the_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A three-backtick line inside a four-backtick fence is content, not the end."""
    text = "````markdown\n```python\nx = 1\n```\n````\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == [
        ("markdown", "```python\nx = 1\n```"),
    ]


def test_a_closer_with_text_after_it_does_not_close_the_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A line like ``` python` is an opener-shaped line, not a closer."""
    text = "```text\n``` python\nbody\n```   \n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == ["``` python\nbody"]


@pytest.mark.parametrize(
    "trailer",
    ["\u00a0", "\x0b", "\x0c", "\x1c", "\u0085", "\u2003", "\u2028", "\u2029", "\u3000"],
    ids=[
        "no-break-space",
        "vertical-tab",
        "form-feed",
        "file-separator",
        "next-line",
        "em-space",
        "line-separator",
        "paragraph-separator",
        "ideographic-space",
    ],
)
def test_a_closer_followed_by_anything_but_a_space_or_tab_does_not_close_the_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trailer: str,
) -> None:
    """Only a space or a tab may follow a closing fence.

    Every character here is whitespace to ``str.strip()``, and several of them
    also end a line for ``str.splitlines()``; none of them is whitespace or a
    line ending to CommonMark, so the second line below is content of the block
    and the last line closes it. markdown-it renders the same document as one
    block.
    """
    text = f"```text\nbody\n```{trailer}\nmore\n```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [f"body\n```{trailer}\nmore"]


@pytest.mark.parametrize(
    "trailer", [" ", "\t", " \t ", "     "], ids=["space", "tab", "mixed", "wide"]
)
def test_a_closer_followed_only_by_spaces_or_tabs_closes_the_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trailer: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, f"```text\nbody\n```{trailer}\nmore\n")

    assert [block.body for _, block in found] == ["body"]


@pytest.mark.parametrize(
    ("opener", "expected_info"),
    [
        ("```python", "python"),
        ("```  python  ", "python"),
        ("``` python title=example.py", "python title=example.py"),
        ("```", ""),
        ("~~~ bash", "bash"),
        ("~~~ note `with backticks`", "note `with backticks`"),
    ],
    ids=["plain", "padded", "attributes", "bare", "tilde", "tilde-with-backticks"],
)
def test_the_info_string_is_captured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
    expected_info: str,
) -> None:
    fence = opener.strip()[0] * 3

    found = _blocks(monkeypatch, tmp_path, f"{opener}\nbody\n{fence}\n")

    assert [info for info, _ in found] == [expected_info]


@pytest.mark.parametrize(
    "opener",
    [
        "```b&#97;sh",
        "~~~b&amp;sh",
        "```&lt;",
        "``` python title=a&b",
        "```ba\\*sh",
        "~~~ba\\&sh",
        "```py\\thon",
        "> ```b&#97;sh",
    ],
    ids=[
        "decimal-reference",
        "named-reference-in-a-tilde-fence",
        "reference-only",
        "bare-ampersand",
        "escaped-punctuation",
        "escaped-ampersand-in-a-tilde-fence",
        "backslash-before-a-letter",
        "inside-a-blockquote",
    ],
)
def test_an_info_string_with_an_ampersand_or_a_backslash_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
) -> None:
    """CommonMark decodes ``&#97;`` and ``\\*`` in an info string; the scan keeps it as written."""
    with pytest.raises(ValueError, match=r"line 1 opens a fenced block whose info string holds"):
        _blocks(monkeypatch, tmp_path, f"{opener}\nbody\n```\n")


@pytest.mark.parametrize(
    "text",
    [
        "```bash\nab\x00cd\n```\n",
        "```ba\x00sh\nx\n```\n",
        "   ```\n   a\x00\n   ```\n",
        "> ```\n> a\x00\n> ```\n",
        "~~~\nx\n\x00~~~\n",
    ],
    ids=[
        "in-the-body",
        "in-the-info-string",
        "in-an-indented-body",
        "in-a-blockquote",
        "before-a-closer",
    ],
)
def test_a_nul_character_in_a_fenced_block_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    """CommonMark reads NUL as U+FFFD; the scan keeps it, so a gate would check other text."""
    with pytest.raises(ValueError, match=r"holds a NUL character in a fenced block"):
        _blocks(monkeypatch, tmp_path, text)


def test_a_nul_character_outside_a_fenced_block_does_not_stop_the_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    found = _blocks(monkeypatch, tmp_path, "a\x00b\n\n```python\nx = 1\n```\n")

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


@pytest.mark.parametrize(
    ("opener", "expected_info"),
    [
        ("```python\u00a0", "python\u00a0"),
        ("```\u00a0python", "\u00a0python"),
        ("```\u00a0", "\u00a0"),
        ("~~~bash\x0c", "bash\x0c"),
    ],
    ids=["trailing-no-break-space", "leading-no-break-space", "only-no-break-space", "form-feed"],
)
def test_a_character_that_is_not_a_space_or_tab_stays_in_the_info_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
    expected_info: str,
) -> None:
    """The info string is trimmed of spaces and tabs only.

    The line still opens a block (markdown-it agrees), and the info string is
    not the bare or plain-language one a gate routes on, so a gate that
    classifies blocks by info string sees a block it does not know.
    """
    fence = opener[0] * 3

    found = _blocks(monkeypatch, tmp_path, f"{opener}\nbody\n{fence}\n")

    assert [(info, block.body) for info, block in found] == [(expected_info, "body")]


def test_a_backtick_in_a_backtick_fence_info_string_is_not_an_opener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``` a`b is inline code in a paragraph, not the start of a code block."""
    found = _blocks(monkeypatch, tmp_path, "``` a`b\n\nprose\n\n```python\nx = 1\n```\n")

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


@pytest.mark.parametrize("line", ["``", "~~", "`", "``python", "text ```"])
def test_a_line_that_is_not_a_run_of_three_fence_characters_does_not_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line: str,
) -> None:
    assert _blocks(monkeypatch, tmp_path, f"{line}\nbody\n") == []


def test_block_positions_and_indentation_are_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The body starts on the line after its opener and is dedented to the fence."""
    text = "prose\n\n  ```python\n  if x:\n      y = 1\n  ```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert len(found) == 1
    info, block = found[0]
    assert info == "python"
    assert block.start_line == 4
    assert block.body == "if x:\n    y = 1"


@pytest.mark.parametrize(
    "character",
    ["\u00a0", "\u2003", "\u3000", "\u202f", "\u1680", "\x0c", "\x0b", "\x1c", "\x85", "\u2028"],
    ids=[
        "no-break-space",
        "em-space",
        "ideographic-space",
        "narrow-no-break-space",
        "ogham-space-mark",
        "form-feed",
        "vertical-tab",
        "file-separator",
        "next-line",
        "line-separator",
    ],
)
def test_a_whitespace_character_that_is_not_a_space_stays_in_a_dedented_body_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    character: str,
) -> None:
    """The dedent removes spaces only.

    ``str.strip()`` would also remove these, so an edited block would read as
    the text a registered digest was taken from.
    """
    text = f"   ```bash\n{character}pip install engrava\n {character}x\n{character}\n   ```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == [
        ("bash", f"{character}pip install engrava\n{character}x\n{character}")
    ]


# ---------------------------------------------------------------------------
# Blockquote-prefixed fences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap", ["", " ", "  ", "   "])
def test_a_blockquoted_fence_with_up_to_three_spaces_after_the_marker_opens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gap: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, f">{gap}```python\n>{gap}x = 1\n>{gap}```\n")

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


def test_a_blockquoted_fence_indented_four_spaces_past_the_marker_does_not_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One space belongs to the marker, so five leaves four: an indented code block."""
    assert _blocks(monkeypatch, tmp_path, ">     ```python\n>     x = 1\n>     ```\n") == []


@pytest.mark.parametrize(
    "marker_line",
    [
        pytest.param(">\t```python", id="tab-after-marker"),
        pytest.param("> \t```python", id="space-then-tab"),
        pytest.param("   >\t```python", id="marker-indented-three"),
    ],
)
def test_a_tab_after_the_blockquote_marker_is_part_of_the_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    marker_line: str,
) -> None:
    """A tab runs to the next multiple of four columns; one column of it is the marker's space."""
    closer = marker_line.replace("python", "")
    found = _blocks(monkeypatch, tmp_path, f"{marker_line}\n{closer}\n")

    assert [(info, block.body) for info, block in found] == [("python", "")]


def test_two_tabs_after_the_blockquote_marker_leave_an_indented_code_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The first tab spans three columns and the second four, so seven minus one is six."""
    assert _blocks(monkeypatch, tmp_path, ">\t\t```python\n>\t\tx = 1\n>\t\t```\n") == []


@pytest.mark.parametrize(
    "stray",
    [
        pytest.param("```", id="fence"),
        pytest.param("", id="blank"),
        pytest.param("text", id="text"),
        pytest.param("\t> sample", id="tab-before-the-marker"),
    ],
)
def test_a_line_without_the_marker_inside_a_blockquoted_fence_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stray: str,
) -> None:
    """CommonMark ends the blockquote there, so the line is read outside the fence."""
    text = f"> ```markdown\n> sample\n{stray}\n| a |\n```\n```\n"

    with pytest.raises(
        ValueError,
        match=r"line 3 has fewer blockquote markers than the fenced block opened at line 1 ",
    ):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    ("prefix", "opens"),
    [
        pytest.param("> > ", True, id="two-levels"),
        pytest.param(">>", True, id="two-levels-no-space"),
        pytest.param("> > > ", True, id="three-levels"),
        pytest.param(">  > ", True, id="two-spaces-then-nested-marker"),
    ],
)
def test_a_fence_nested_in_several_blockquotes_is_read_with_every_marker_stripped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prefix: str,
    opens: bool,
) -> None:
    text = f"{prefix}```python\n{prefix}x = 1\n{prefix}```\nafter\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == ([("python", "x = 1")] if opens else [])


@pytest.mark.parametrize(
    ("prefix", "opens"),
    [
        pytest.param(">\t> \t", True, id="tab-then-nested-marker"),
        pytest.param(">  > \t", True, id="two-spaces-then-nested-marker"),
        pytest.param("> > \t", False, id="nested-marker-then-tab-past-three"),
    ],
)
def test_a_tab_among_nested_blockquote_markers_is_measured_from_the_start_of_the_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prefix: str,
    opens: bool,
) -> None:
    """The columns a tab spans depend on how many markers precede it."""
    found = _blocks(monkeypatch, tmp_path, f"{prefix}```python\n{prefix}```\nafter\n")

    assert [(info, block.body) for info, block in found] == ([("python", "")] if opens else [])


def test_a_line_with_fewer_markers_than_a_nested_blockquoted_fence_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text = "> > ```markdown\n> > sample\n> ```\n"

    with pytest.raises(ValueError, match=r"line 3 has fewer blockquote markers than the fenced "):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize("prefix", ["    > ", ">     > "], ids=["outer", "nested"])
def test_a_blockquote_marker_indented_four_spaces_is_not_a_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prefix: str,
) -> None:
    """Four spaces before a ``>`` make an indented code block, at either level."""
    assert _blocks(monkeypatch, tmp_path, f"{prefix}```python\n{prefix}x = 1\n{prefix}```\n") == []


def test_a_body_line_may_carry_more_markers_than_its_blockquoted_fence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    found = _blocks(monkeypatch, tmp_path, "> ```text\n> > quoted output\n> ```\n")

    assert [block.body for _, block in found] == ["> quoted output"]


@pytest.mark.parametrize(
    "opener",
    ["- ```markdown", "> - ```markdown", "- > ```markdown", "1. - ```markdown", "- - ```markdown"],
)
def test_a_fence_opened_after_a_list_marker_is_an_error_at_any_depth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
) -> None:
    with pytest.raises(ValueError, match="opens a fence on a list-item line"):
        _blocks(monkeypatch, tmp_path, f"{opener}\n  | a |\n  ```\n")


def test_a_plain_fence_after_a_closed_blockquoted_fence_opens_its_own_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text = "> ```python\n> x = 1\n> ```\n```text\ny\n```\n"

    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == [("python", "x = 1"), ("text", "y")]


def test_a_plain_fence_keeps_a_literal_angle_bracket_in_its_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    found = _blocks(monkeypatch, tmp_path, "```text\n> quoted output\n```\n")

    assert [block.body for _, block in found] == ["> quoted output"]


# ---------------------------------------------------------------------------
# Unclosed fences
# ---------------------------------------------------------------------------


def test_a_fence_that_is_never_closed_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"opened at line 2 \(```python\) is never closed"):
        _blocks(monkeypatch, tmp_path, "prose\n```python\nx = 1\n")


def test_a_shorter_closer_leaves_a_longer_fence_unclosed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="never closed"):
        _blocks(monkeypatch, tmp_path, "````python\nx = 1\n```\n")


# ---------------------------------------------------------------------------
# A fence opened on a list-marker line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opener",
    [
        "- ```markdown",
        "* ~~~",
        "+ ```",
        "1. ```python",
        "12) ~~~text",
        "  - ```markdown",
        "    - ```markdown",
        "-\t```markdown",
        "> - ```markdown",
    ],
)
def test_a_fence_opened_on_a_list_marker_line_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
) -> None:
    """The scan does not model list items, so it refuses the construct.

    Read as prose, the closer of such a fence looks like an opener, and a stray
    fence later in the file balances it: a table between them is code to a
    renderer and text to the scan. The trailing pair below is that balanced
    shape, so the error asserted is the list-marker one, not "never closed".
    """
    text = f"{opener}\n  | a |\n  ```\n\nprose\n```\n"

    with pytest.raises(ValueError, match=r"page\.md: line 1 opens a fence on a list-item line"):
        _blocks(monkeypatch, tmp_path, text)


def test_lines_outside_fences_rejects_a_fence_opened_on_a_list_marker_line() -> None:
    lines = ["- ```markdown", "  | a |", "  ```", "", "```"]

    with pytest.raises(ValueError, match=r"page\.md: line 1 opens a fence on a list-item line"):
        lines_outside_fences(lines, "page.md")


def test_a_fence_on_its_own_line_under_a_list_item_is_still_a_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    found = _blocks(monkeypatch, tmp_path, "- an item\n\n  ```python\n  x = 1\n  ```\n")

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


@pytest.mark.parametrize(
    "line",
    ["- use ``` to open a fence", "-```python", "1.```python", "- `x`"],
)
def test_a_list_marker_line_that_does_not_open_a_fence_is_prose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line: str,
) -> None:
    assert _blocks(monkeypatch, tmp_path, f"{line}\n") == []


# ---------------------------------------------------------------------------
# What the scan does not model: HTML blocks and the indentation of list items
# ---------------------------------------------------------------------------

_HTML_WINDOW_ERROR = r"page\.md: line \d+ looks like a fence while an HTML block may be open"
_INDENTED_FENCE_ERROR = r"page\.md: line \d+ looks like a fence but is indented four or more"
_BODY_TAB_ERROR = r"page\.md: line 2 has a tab inside the indentation of the fenced block opened"
_BODY_SHIFT_ERROR = r"page\.md: line \d+ is indented less than the fenced block opened at line"
_BODY_MARKER_TAB_ERROR = (
    r"page\.md: line 2 has a tab among its blockquote markers and leading whitespace"
)
_OPENER_MARKERS_ERROR = (
    r"page\.md: line 3 opens a blockquoted fence whose blockquote markers are indented"
)


@pytest.mark.parametrize(
    "start",
    [
        "<div>",
        "</div>",
        "<custom-tag>",
        "<script>",
        "<PRE>",
        "<style media=screen>",
        "<textarea>",
        "<!-- a comment",
        "<?php",
        "<!DOCTYPE html",
        "<![CDATA[",
        "  <div>",
        "> <div>",
        "- <div>",
    ],
)
def test_a_fence_like_line_right_after_an_html_block_start_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    start: str,
) -> None:
    """A fence-like line inside an HTML block is HTML text; the scan does not follow HTML blocks."""
    with pytest.raises(ValueError, match=_HTML_WINDOW_ERROR):
        _blocks(monkeypatch, tmp_path, f"{start}\n```python\nx = 1\n```\n")


@pytest.mark.parametrize(
    "text",
    [
        "<div>\n\n```python\nx = 1\n```\n",
        "<!-- a comment -->\n```python\nx = 1\n```\n",
        "<!-->\n```python\nx = 1\n```\n",
        "<script>\n</script>\n\n```python\nx = 1\n```\n",
        "<pre>x</pre>\n```python\nx = 1\n```\n",
        "<!--\ntext\n-->\n```python\nx = 1\n```\n",
        "<?php echo 1; ?>\n```python\nx = 1\n```\n",
        "<!DOCTYPE html>\n```python\nx = 1\n```\n",
        "<![CDATA[ x ]]>\n```python\nx = 1\n```\n",
        "<textarea>\n</textarea>\n\n```python\nx = 1\n```\n",
        "a < b and c <3\n```python\nx = 1\n```\n",
        "prose\n```python\nx = 1\n```\n<div>\n",
    ],
)
def test_a_fence_after_an_html_block_has_ended_is_a_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, text)

    assert [(info, block.body) for info, block in found] == [("python", "x = 1")]


@pytest.mark.parametrize(
    "text",
    [
        "<script>\n\n```python\nx = 1\n```\n",
        "<!--\n\n```python\nx = 1\n```\n",
        "<textarea>\n</textarea>\n```python\nx = 1\n```\n",
        "<textarea>\nx</textarea>\n```python\nx = 1\n```\n",
        "<style media=screen>\n\n```python\nx = 1\n```\n",
        "<script\n\n```python\nx = 1\n```\n",
        "<div>\n<script>\n\n```python\nx = 1\n```\n",
        "<!--\n<script>\n-->\n```python\nx = 1\n```\n",
    ],
)
def test_a_blank_line_does_not_end_an_html_block_that_ends_at_a_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    with pytest.raises(ValueError, match=_HTML_WINDOW_ERROR):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    "text",
    [
        "- an item\n\n    ```python\n    x = 1\n    ```\n",
        "1. an item\n\n    ~~~\n    x = 1\n    ~~~\n",
        "- an item\n\n\t```python\n\tx = 1\n\t```\n",
        "> - an item\n>\n>     ```python\n>     x = 1\n>     ```\n",
        "- an item\n\ntext\n\n    ```python\n    x = 1\n    ```\n",
    ],
)
def test_a_fence_like_line_indented_four_columns_or_by_a_tab_after_a_list_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    """A list item may make a fence of a line indented that far."""
    with pytest.raises(ValueError, match=_INDENTED_FENCE_ERROR):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    "text",
    [
        "    ```python\n    x = 1\n    ```\n- an item\n",
        "\t```python\n\tx = 1\n\t```\n",
        "intro\n\n    ```python\n    x = 1\n    ```\n",
    ],
)
def test_a_fence_like_line_indented_four_columns_or_by_a_tab_without_a_list_is_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    assert _blocks(monkeypatch, tmp_path, text) == []


def test_lines_outside_fences_rejects_a_fence_like_line_after_an_html_block_start() -> None:
    with pytest.raises(ValueError, match=r"page\.md: line 2 looks like a fence while an HTML"):
        lines_outside_fences(["<div>", "```python", "x = 1", "```"], "page.md")


def test_lines_outside_fences_rejects_an_indented_fence_like_line_after_a_list() -> None:
    lines = ["- an item", "", "    ```python", "    x = 1", "    ```"]

    with pytest.raises(ValueError, match=r"page\.md: line 3 looks like a fence but is indented"):
        lines_outside_fences(lines, "page.md")


@pytest.mark.parametrize(
    "text",
    [
        "- an item\n\n  ```python\nx = 1\n  ```\n",
        "- an item\n\n  ```python\n x = 1\n  ```\n",
        "- an item\n\n  ```python\n    ```\n  ```\n",
        "- an item\n\n  ```python\n     ```  \n  ```\n",
        "- an item\n\n```python\n    ```\n```\n",
        "- an item\n\n```python\n\t```\n```\n",
    ],
)
def test_a_body_line_that_a_list_item_may_move_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    """A list item may end the block at this line, or read the line as its closer."""
    with pytest.raises(ValueError, match=_BODY_SHIFT_ERROR):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("  ```python\nx = 1\n  ```\n", "x = 1"),
        ("  ```python\n    ```\n  ```\n", "  ```"),
        ("- an item\n\n  ```python\n  x = 1\n  ```\n", "x = 1"),
        ("- an item\n\n  ```python\n\n  x = 1\n\n  ```\n", "\nx = 1\n"),
        ("- an item\n\n  ```python\n    x = 1\n  ```\n", "  x = 1"),
        ("- an item\n\n```python\nx = 1\n```\n", "x = 1"),
        ("- an item\n\n```python\n\tx = 1\n```\n", "\tx = 1"),
    ],
)
def test_a_body_line_no_list_item_can_move_is_read_as_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
    body: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [body]


@pytest.mark.parametrize(
    "text",
    [
        "  ```python\n\tx = 1\n  ```\n",
        "  ```python\n \t\n  ```\n",
        "   ```python\n\t\n   ```\n",
    ],
)
def test_a_tab_inside_the_indentation_of_an_indented_fences_body_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    """Removing the fence's indentation from such a line depends on tab stops the scan skips."""
    with pytest.raises(ValueError, match=_BODY_TAB_ERROR):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("```python\n\tx = 1\n```\n", "\tx = 1"),
        (" ```python\n \tx = 1\n ```\n", "\tx = 1"),
        ("  ```python\n  \tx = 1\n  ```\n", "\tx = 1"),
        ("  ```python\n\n  x = 1\n  ```\n", "\nx = 1"),
    ],
)
def test_a_tab_outside_the_indentation_of_a_fences_body_is_kept(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
    body: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [body]


@pytest.mark.parametrize(
    "text",
    [
        "- an item\n\n  ```python\n     \n  ```\n",
        "- an item\n\n  ```python\n   \n  ```\n",
        "- an item\n\n```python\n   \n```\n",
        "- an item\n\n  ```python\n  x = 1\n     \n  ```\n",
    ],
)
def test_a_whitespace_only_body_line_beyond_the_fences_indentation_is_an_error_after_a_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    """A list item may drop the whitespace of such a line, which the scan keeps."""
    with pytest.raises(ValueError, match=_BODY_SHIFT_ERROR):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("- an item\n\n  ```python\n  \n  x = 1\n  ```\n", "\nx = 1"),
        ("- an item\n\n  ```python\n \n  x = 1\n  ```\n", "\nx = 1"),
        ("- an item\n\n```python\n\nx = 1\n```\n", "\nx = 1"),
        ("```python\n   \nx = 1\n```\n", "   \nx = 1"),
        ("  ```python\n     \n  ```\n", "   "),
    ],
)
def test_a_whitespace_only_body_line_no_list_item_can_change_is_read_as_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
    body: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [body]


@pytest.mark.parametrize(
    "opener",
    ["  > ```python", " > ```python", "   > ```python", ">   > ```python", ">\t> ```python"],
)
def test_a_blockquoted_opener_with_indented_markers_is_an_error_after_a_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener: str,
) -> None:
    """A list item may hold the blockquote, and where that item ends is not modelled."""
    with pytest.raises(ValueError, match=_OPENER_MARKERS_ERROR):
        _blocks(monkeypatch, tmp_path, f"- an item\n\n{opener}\nx\n```\n")


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("- an item\n\n> ```python\n> x = 1\n> ```\n", "x = 1"),
        ("- an item\n\n>```python\n>x = 1\n>```\n", "x = 1"),
        ("- an item\n\n> > ```python\n> > x = 1\n> > ```\n", "x = 1"),
        ("- an item\n\n>> ```python\n>> x = 1\n>> ```\n", "x = 1"),
        ("  > ```python\n  > x = 1\n  > ```\n", "x = 1"),
        (">   > ```python\n>   > x = 1\n>   > ```\n", "x = 1"),
    ],
)
def test_a_blockquoted_fence_no_list_item_can_hold_is_read_as_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
    body: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [body]


@pytest.mark.parametrize(
    "text",
    [
        "> ```python\n>\tx = 1\n> ```\n",
        "> ```python\n> \tx = 1\n> ```\n",
        "> ```python\n>  \tx = 1\n> ```\n",
        "> ```python\n>\t\t\n> ```\n",
        "> ```python\n>\t\n> ```\n",
        ">\t```python\n>\tx = 1\n>\t```\n",
        "> > ```python\n> >\tx = 1\n> > ```\n",
    ],
)
def test_a_tab_among_the_markers_and_leading_whitespace_of_a_blockquoted_body_line_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
) -> None:
    """How much of that whitespace is the block's text depends on tab stops the scan lacks."""
    with pytest.raises(ValueError, match=_BODY_MARKER_TAB_ERROR):
        _blocks(monkeypatch, tmp_path, text)


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ("> ```python\n> x =\t1\n> ```\n", "x =\t1"),
        ("> ```text\n> a > b\tc\n> ```\n", "a > b\tc"),
        ("```python\n\tx = 1\n```\n", "\tx = 1"),
        ("> ```python\n> x = 1\t\n> ```\n", "x = 1\t"),
    ],
)
def test_a_tab_after_the_text_of_a_blockquoted_body_line_is_kept(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    text: str,
    body: str,
) -> None:
    found = _blocks(monkeypatch, tmp_path, text)

    assert [block.body for _, block in found] == [body]


def test_lines_outside_fences_rejects_a_tabbed_body_line_of_a_blockquoted_fence() -> None:
    lines = ["> ```python", ">\tx = 1", "> ```"]

    with pytest.raises(ValueError, match=r"page\.md: line 2 has a tab among its blockquote"):
        lines_outside_fences(lines, "page.md")


# ---------------------------------------------------------------------------
# lines_outside_fences
# ---------------------------------------------------------------------------


def test_lines_outside_fences_blanks_each_block_and_keeps_every_other_line() -> None:
    lines = [
        "## Heading",
        "```markdown",
        "| Exception | Base |",
        "```",
        "prose",
        "~~~",
        "## Fenced heading",
        "~~~",
        "| Exception | Base |",
    ]

    assert lines_outside_fences(lines, "page.md") == [
        "## Heading",
        "",
        "",
        "",
        "prose",
        "",
        "",
        "",
        "| Exception | Base |",
    ]


def test_lines_outside_fences_leaves_a_document_without_fences_unchanged() -> None:
    lines = ["# Title", "", "| a | b |", "|---|---|", "    indented code", "    ```not a fence"]

    assert lines_outside_fences(lines, "page.md") == lines


def test_lines_outside_fences_does_not_mutate_its_input() -> None:
    lines = ["```", "x", "```"]

    lines_outside_fences(lines, "page.md")

    assert lines == ["```", "x", "```"]


def test_lines_outside_fences_follows_the_same_indent_rule() -> None:
    """A four-space "opener" hides nothing; a three-space one hides its whole block."""
    lines = ["    ```", "visible", "   ```", "hidden", "```", "visible again"]

    assert lines_outside_fences(lines, "page.md") == [
        "    ```",
        "visible",
        "",
        "",
        "",
        "visible again",
    ]


def test_lines_outside_fences_blanks_a_blockquoted_fence() -> None:
    lines = ["> ```python", "> x = 1", "> ```", "after"]

    assert lines_outside_fences(lines, "page.md") == ["", "", "", "after"]


def test_lines_outside_fences_rejects_a_line_that_ends_a_blockquote_inside_a_fence() -> None:
    lines = ["> ```markdown", "> | a |", "```", "| b |", "```"]

    with pytest.raises(ValueError, match=r"page\.md: line 3 has fewer blockquote markers"):
        lines_outside_fences(lines, "page.md")


def test_lines_outside_fences_rejects_an_unclosed_fence() -> None:
    with pytest.raises(ValueError, match=r"page\.md: a fenced block opened at line 1"):
        lines_outside_fences(["```python", "x = 1"], "page.md")


def test_lines_outside_fences_does_not_close_a_block_at_a_no_break_space_closer() -> None:
    lines = ["```markdown", "| a |", "```\u00a0", "| b |", "```", "after"]

    assert lines_outside_fences(lines, "page.md") == ["", "", "", "", "", "after"]


# ---------------------------------------------------------------------------
# markdown_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "a", "a\n", "a\nb", "\n", "\n\n", "a\n\nb\n\n", "a\r\nb\r\n", "a\rb\r", "a\r\r\nb\n\rc"],
)
def test_markdown_lines_matches_splitlines_when_only_commonmark_line_endings_occur(
    text: str,
) -> None:
    assert markdown_lines(text) == text.splitlines()


@pytest.mark.parametrize(
    "separator",
    ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\u0085", "\u2028", "\u2029"],
    ids=[
        "vertical-tab",
        "form-feed",
        "file-separator",
        "group-separator",
        "record-separator",
        "next-line",
        "line-separator",
        "paragraph-separator",
    ],
)
def test_markdown_lines_keeps_a_character_that_only_splitlines_treats_as_a_line_ending(
    separator: str,
) -> None:
    text = f"a{separator}b\n"

    assert text.splitlines() == ["a", "b"]
    assert markdown_lines(text) == [f"a{separator}b"]


# ---------------------------------------------------------------------------
# Exemption digests
# ---------------------------------------------------------------------------


def _block(body: str) -> CodeBlock:
    return CodeBlock(path=Path("page.md"), rel="page.md", start_line=7, body=body)


def test_block_digest_is_a_stable_sixteen_hex_digit_string() -> None:
    digest = block_digest("pip install engrava")

    assert re.fullmatch(r"[0-9a-f]{16}", digest)
    assert digest == block_digest("pip install engrava")


@pytest.mark.parametrize(
    "edited",
    [
        "pip install engrava\nengrava reindex",
        "pip install engrava ",
        "pip  install engrava",
        "Pip install engrava",
        "",
    ],
)
def test_block_digest_differs_for_an_edited_text(edited: str) -> None:
    assert block_digest(edited) != block_digest("pip install engrava")


def test_exemption_digest_problems_is_empty_when_every_digest_matches() -> None:
    block = _block("pip install engrava")

    assert exemption_digest_problems("REGISTRY", [(block, block_digest(block.body))]) == []


def test_exemption_digest_problems_names_the_block_and_the_digest_to_register() -> None:
    block = _block("pip install engrava\nengrava reindex")
    registered = block_digest("pip install engrava")

    problems = exemption_digest_problems("EXEMPT_BASH_BLOCKS", [(block, registered)])

    assert len(problems) == 1
    message = problems[0]
    assert message.startswith("page.md:7:")
    assert registered in message
    assert block_digest(block.body) in message
    assert "EXEMPT_BASH_BLOCKS" in message
    assert "checked again" in message


def test_exemption_digest_problems_reports_only_the_stale_entries() -> None:
    fresh = _block("make install")
    stale = CodeBlock(path=fresh.path, rel="other.md", start_line=3, body="make install && rm x")

    problems = exemption_digest_problems(
        "REGISTRY",
        [(fresh, block_digest(fresh.body)), (stale, block_digest("make install"))],
    )

    assert len(problems) == 1
    assert problems[0].startswith("other.md:3:")
