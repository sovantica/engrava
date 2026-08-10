"""Shared helpers for the documentation-example test suite.

The documentation tests treat the project's Markdown files (``README.md``
and everything under ``docs/``) as a source of executable truth. These
helpers locate the Markdown files and extract their fenced ``python``
code blocks so the individual test modules can compile, scan, or execute
them.

A fenced block is recognised by an opening line whose first non-space
token is ```` ```python ```` (optionally followed by extra info-string
words, which renderers ignore) and a closing line that is exactly
```` ``` ````. Indented blocks are supported; the captured body is
dedented to the fence's indentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# tests/docs/_md_blocks.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"

_CLOSING_FENCE = "```"


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


def extract_fenced_blocks(path: Path, language: str) -> list[CodeBlock]:
    """Extract every fenced block of one info-string language from a Markdown file.

    ``python`` blocks are the executable-example surface; other languages carry
    documented *output* (a ``text`` block showing what an example prints), which
    a test can compare against a real run so the page cannot promise a result it
    does not produce.

    Args:
        path: The Markdown file to scan.
        language: The fence info-string language, e.g. ``"python"`` or ``"text"``.

    Returns:
        A list of :class:`CodeBlock` in document order. Empty when the file
        contains no fence of that language.

    """
    fence = f"```{language}"
    rel = path.relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()

    blocks: list[CodeBlock] = []
    in_block = False
    indent = 0
    body_lines: list[str] = []
    body_start = 0

    for index, raw in enumerate(lines):
        stripped = raw.lstrip()
        if not in_block:
            if stripped.startswith(fence):
                in_block = True
                indent = len(raw) - len(stripped)
                body_lines = []
                body_start = index + 2  # 1-based, first line after the fence
            continue
        # Inside a block: a line that is exactly the closing fence ends it.
        if stripped == _CLOSING_FENCE:
            body = "\n".join(_dedent(line, indent) for line in body_lines)
            blocks.append(
                CodeBlock(path=path, rel=rel, start_line=body_start, body=body),
            )
            in_block = False
            continue
        body_lines.append(raw)

    return blocks


def all_python_blocks() -> list[CodeBlock]:
    """Return every ``python`` code block across all documentation files."""
    blocks: list[CodeBlock] = []
    for path in markdown_files():
        blocks.extend(extract_python_blocks(path))
    return blocks


def _dedent(line: str, indent: int) -> str:
    """Strip up to ``indent`` leading spaces from a captured body line."""
    stripped = line[:indent]
    if stripped.strip() == "":
        return line[indent:]
    return line.lstrip()
