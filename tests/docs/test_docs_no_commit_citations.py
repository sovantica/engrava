"""Keeps commit-hash citations out of public prose.

**The rule.** Public prose describes behaviour by release (what ``0.6.0`` did,
what ``0.7.0`` does) and does not cite a commit by hash. A reader arrives from a
release, cannot resolve a bare hash, and learns nothing from it. To fix a
failure here, state the behaviour and delete the hash.

**What is flagged.** In ``docs/**/*.md`` prose, in the comments and docstrings of
every ``.py`` file under ``src/``, and in the comments and docstrings of every
``.py`` file under ``tests/``: an inline-code token (one or two backticks each
side), or a token directly following the word "commit" (optionally
backtick-quoted, on the same or the next line), made of 7 to 40 lowercase hex
characters with at least one digit and at least one letter.

**What is not covered.**

* Fenced code blocks in Markdown (their fences included).
* Markdown files outside ``docs/``, and anything that is not a ``.py`` file
  (``scripts/``, ``examples/``, SQL or YAML resources).
* String literals in Python, including test fixtures and assertion messages:
  only comments and docstrings are read.
* A hash written without backticks that does not directly follow the word
  "commit": "commit hash 1a2b3c4" and "commit: 1a2b3c4" are not flagged.
* Id fragments. A UUID fragment cannot be told from a short hash by shape, so
  one in backticks *is* flagged; write it in a fenced block, or without
  backticks. A full UUID is not matched.

A citation that is a legitimate, checkable revision of this repository goes in
``_ALLOWED`` with its reason, by exact path and token. An entry whose citation is
gone fails ``test_every_allowed_citation_is_still_present``, so the list cannot
outlive what it excuses.
"""

from __future__ import annotations

import ast
import functools
import io
import re
import sys
import tokenize
from typing import TYPE_CHECKING

import pytest

from tests.docs._md_blocks import DOCS_DIR, REPO_ROOT, lines_outside_fences, markdown_lines

if TYPE_CHECKING:
    from pathlib import Path

_HEX_RUN = r"[0-9a-f]{7,40}"

# A token wrapped in one or two backticks, and nothing else inside them.
_QUOTED = re.compile(rf"(?<!`)`{{1,2}}({_HEX_RUN})`{{1,2}}(?!`)")
# A token after the word "commit": separated by blanks or by one line break,
# so a citation wrapped onto the next line is still seen, but a blank line
# (or a fenced block blanked out between them) breaks the link.
_AFTER_COMMIT = re.compile(
    rf"(?i:\bcommits?)(?:[ \t]+|[ \t]*\n[ \t]*)`{{0,2}}({_HEX_RUN})(?![0-9A-Za-z_-])"
)

_BENCHMARK_REVISION = (
    "a revision of this repository that a published measurement names so it can be reproduced"
)

#: ``(path relative to the repository root, token)`` -> why that citation stays.
_ALLOWED: dict[tuple[str, str], str] = {
    ("docs/benchmarks.md", "88b535b"): _BENCHMARK_REVISION,
    ("docs/benchmarks.md", "2918e38"): _BENCHMARK_REVISION,
    ("docs/benchmarks.md", "1033a2e"): _BENCHMARK_REVISION,
    (
        "tests/test_main_carries_the_released_tag.py",
        "8c044e2",
    ): "a commit of the public repository's main, named with its pull-request title",
}


def _has_digit_and_letter(token: str) -> bool:
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def find_citations(text: str) -> list[tuple[int, str]]:
    """Return ``(line, token)`` for every commit-hash citation in ``text``."""
    found: set[tuple[int, str]] = set()
    for pattern in (_QUOTED, _AFTER_COMMIT):
        for match in pattern.finditer(text):
            token = match.group(1)
            if _has_digit_and_letter(token):
                found.add((text.count("\n", 0, match.start(1)) + 1, token))
    return sorted(found)


def python_prose_lines(source: str) -> list[str]:
    """Return ``source`` reduced to its comments and docstrings, one entry per line."""
    lines = source.split("\n")
    view = [""] * len(lines)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            view[token.start[0] - 1] = token.string.removeprefix("#")
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, scopes) or not node.body:
            continue
        first = node.body[0]
        is_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        if is_docstring:
            for number in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                view[number - 1] = lines[number - 1]
    return view


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _docs_citations() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = _relative(path)
        # A document the scan cannot read the way CommonMark does raises
        # ValueError here rather than being skipped: this function has no
        # try/except around it, so the error reaches the caller and fails the
        # guard on that file instead of silently passing it.
        lines = lines_outside_fences(markdown_lines(path.read_text(encoding="utf-8")), rel)
        text = "\n".join(lines)
        found.extend((rel, line, token) for line, token in find_citations(text))
    return found


def _python_citations(tree: str) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted((REPO_ROOT / tree).rglob("*.py")):
        text = "\n".join(python_prose_lines(path.read_text(encoding="utf-8")))
        found.extend((_relative(path), line, token) for line, token in find_citations(text))
    return found


@functools.cache
def _citations(tree: str) -> tuple[tuple[str, int, str], ...]:
    if tree == "docs":
        return tuple(_docs_citations())
    return tuple(_python_citations(tree))


_TREES = ("docs", "src", "tests")


@pytest.mark.parametrize("tree", _TREES)
def test_no_commit_hash_is_cited(tree: str) -> None:
    offenders = [
        f"{path}:{line}: `{token}`"
        for path, line, token in _citations(tree)
        if (path, token) not in _ALLOWED
    ]
    assert not offenders, (
        "Public prose cites a commit by hash. Describe the behaviour by release "
        "(0.6.0 versus 0.7.0) and delete the hash:\n" + "\n".join(offenders)
    )


def test_every_allowed_citation_is_still_present() -> None:
    present = {(path, token) for tree in _TREES for path, _, token in _citations(tree)}
    stale = sorted(set(_ALLOWED) - present)
    assert not stale, f"Remove these entries from _ALLOWED, their citations are gone: {stale}"


# ---------------------------------------------------------------------------
# The scanner itself, on synthetic text.
# ---------------------------------------------------------------------------

_SHORT = "1a2b3c4"
_LONG = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    "text",
    [
        f"Fixed in `{_SHORT}` last week.",
        f"introduced by ``{_SHORT}``, which",
        f"see commit {_SHORT} for the change",
        f"see Commit `{_SHORT}` for the change",
        f"see commit\n  {_SHORT} for the change",
        f"the `{_LONG}` revision",
    ],
)
def test_scanner_flags_a_cited_commit(text: str) -> None:
    assert find_citations(text), text


@pytest.mark.parametrize(
    "text",
    [
        "the count is `1234567` rows",
        "the word `deadbeef` alone",
        "too short: `1a2b3c`",
        f"too long: `{_LONG}0`",
        "uppercase: `1A2B3C4D`",
        f"unquoted, no commit: id {_SHORT} in a sentence",
        "a full id `550e8400-e29b-41d4-a716-446655440000` in backticks",
        f"commit message {_SHORT}",
        f"commit hash {_SHORT}",
        f"commit: {_SHORT}",
        f"see commit\n\n{_SHORT} after a blank line",
    ],
)
def test_scanner_ignores_what_is_not_a_cited_commit(text: str) -> None:
    assert not find_citations(text), text


def test_scanner_reports_the_line_of_the_token() -> None:
    assert find_citations(f"one\ntwo `{_SHORT}`\nthree commit\n{_SHORT}") == [
        (2, _SHORT),
        (4, _SHORT),
    ]


def test_a_quoted_id_fragment_is_flagged_because_it_looks_like_a_short_hash() -> None:
    assert find_citations("row `091aa106` was updated")
    assert not find_citations("row 091aa106 was updated")


def test_a_hash_shaped_token_inside_a_fenced_block_is_not_flagged() -> None:
    source = f"before\n```text\nin `{_SHORT}`\n```\nafter `{_SHORT}`\n"
    lines = lines_outside_fences(markdown_lines(source), "x.md")
    assert find_citations("\n".join(lines)) == [(5, _SHORT)]


def _point_the_docs_scan_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "REPO_ROOT", root)
    monkeypatch.setattr(module, "DOCS_DIR", root / "docs")


def test_the_docs_scan_reports_a_prose_citation_and_skips_a_fenced_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text(
        f"prose\n```text\nin `{_SHORT}`\n```\nafter `{_SHORT}`\n", encoding="utf-8"
    )
    _point_the_docs_scan_at(monkeypatch, tmp_path)
    assert _docs_citations() == [("docs/x.md", 5, _SHORT)]


def test_a_document_the_fence_scan_refuses_fails_the_guard_rather_than_skipping_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The docs guard must not silently pass a file it cannot scan.

    A fence that is never closed makes the fence scan raise ``ValueError``. The
    docs scan has to let that error out, so the guard fails on the file, instead
    of treating the file as having no citations.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text(
        f"prose\n```python\nunclosed fence with `{_SHORT}` inside\n", encoding="utf-8"
    )
    _point_the_docs_scan_at(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="never closed"):
        _docs_citations()


def test_python_scan_reads_comments_and_docstrings_only() -> None:
    source = (
        f'"""Module docstring cites `{_SHORT}`."""\n'
        "\n"
        "\n"
        "class Klass:\n"
        f'    """Class docstring: commit {_SHORT}."""\n'
        "\n"
        "    async def method(self) -> str:\n"
        f'        """Method docstring: `{_SHORT}`."""\n'
        f"        # a comment that cites `{_SHORT}`\n"
        f'        return "`{_SHORT}` in a string literal"\n'
    )
    text = "\n".join(python_prose_lines(source))
    assert [line for line, _ in find_citations(text)] == [1, 5, 8, 9]
