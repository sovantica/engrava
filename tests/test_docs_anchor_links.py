"""Confirms every cross-file ``docs/*.md`` anchor link resolves to a real heading.

Scope is narrow and deliberate: only a Markdown link of the form
``](other.md#fragment)`` inside a top-level ``docs/*.md`` file, where
``other.md`` names another file that actually exists directly inside
``docs/``, is checked. A same-page ``#fragment`` link, a link into a file
outside ``docs/`` (e.g. the top-level ``README.md``), and a link with no
``#fragment`` are all out of scope -- this closes exactly the hole a stale
or mistyped cross-file anchor left, not general link-checking.

The check re-derives GitHub's heading-anchor slug for every heading in the
target file (skipping headings that appear inside fenced code blocks, since
a Python or bash comment starting with ``#`` is not a heading) and asserts
the link's fragment matches one of them. It does not validate Markdown
table structure or prose correctness, and it does not disambiguate
duplicate headings the way GitHub does (appending ``-1``, ``-2``, ...) --
no link in this documentation set currently targets a suffixed anchor, so
that disambiguation is out of scope until one does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

_CROSS_FILE_ANCHOR_LINK = re.compile(r"\]\(([A-Za-z0-9_.-]+\.md)#([A-Za-z0-9_-]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(\S.*)$")
_FENCE_PREFIXES = ("```", "~~~")


def _github_slug(heading_text: str) -> str:
    """Reproduce GitHub's heading-anchor slug algorithm.

    Lowercase, then spaces become hyphens, then anything outside
    ``[a-z0-9_-]`` is dropped outright (not replaced by a hyphen). Verified
    below against two known cases -- including a punctuated, multi-word one
    -- before being trusted against the real documentation set.
    """
    text = heading_text.strip().lower()
    text = text.replace(" ", "-")
    return re.sub(r"[^a-z0-9_-]", "", text)


def test_github_slug_of_a_plain_heading() -> None:
    assert _github_slug("Exceptions") == "exceptions"


def test_github_slug_of_a_punctuated_multiword_heading() -> None:
    assert (
        _github_slug("Optimistic concurrency and `StaleDataError`")
        == "optimistic-concurrency-and-staledataerror"
    )


def _headings_in(path: Path) -> set[str]:
    """Return the slug of every heading in ``path``, skipping fenced code blocks."""
    slugs: set[str] = set()
    in_fence = False
    fence_marker = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if not in_fence and stripped[:3] in _FENCE_PREFIXES:
            in_fence = True
            fence_marker = stripped[:3]
            continue
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        match = _HEADING.match(line)
        if match:
            slugs.add(_github_slug(match.group(1)))
    return slugs


def _cross_file_doc_anchor_links() -> list[tuple[str, int, str, str]]:
    """Return ``(source_name, line_no, target_name, fragment)`` for every in-scope link.

    Every link matching the in-scope shape is returned, whether or not
    ``target_name`` names a file that actually exists in ``docs/`` --
    dropping a link here because its target is missing would hide exactly
    the defect (a stale or mistyped file name) this module exists to catch.
    """
    links: list[tuple[str, int, str, str]] = []
    for source_path in sorted(DOCS_DIR.glob("*.md")):
        for line_no, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for target_name, fragment in _CROSS_FILE_ANCHOR_LINK.findall(line):
                links.append((source_path.name, line_no, target_name, fragment))
    return links


@pytest.mark.parametrize(
    ("source_name", "line_no", "target_name", "fragment"),
    _cross_file_doc_anchor_links(),
)
def test_cross_file_doc_anchor_resolves(
    source_name: str, line_no: int, target_name: str, fragment: str
) -> None:
    target_path = DOCS_DIR / target_name
    assert target_path.is_file(), (
        f"docs/{source_name}:{line_no} links to {target_name}, which does not exist in docs/"
    )
    target_headings = _headings_in(target_path)
    assert fragment in target_headings, (
        f"docs/{source_name}:{line_no} links to {target_name}#{fragment}, but no "
        f"heading in docs/{target_name} slugs to {fragment!r}"
    )
