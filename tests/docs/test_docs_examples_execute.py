"""Layer 1 of the documentation-example tests — execute self-contained blocks.

Some documentation code blocks are complete, runnable scripts (they import
what they use and drive themselves via ``asyncio.run``). This module executes
each such block exactly as a reader would — by writing it to a temp file and
running it in a subprocess against the installed ``engrava`` — and asserts a
clean exit. This is the strongest guarantee: the published snippet *runs*.

Fragment blocks (the majority — they assume an existing ``store``/``conn`` or
show a class definition) are out of scope here; they are covered by the
compile + phantom-API guards in ``test_docs_examples_compile.py`` and by the
behaviour tests in ``test_docs_examples_behavior.py``.

The set of self-contained blocks is pinned by an **explicit allowlist** keyed
on ``file:line`` so the doc surface stays clean (no special fence
annotations leak to GitHub / engrava.ai). When you move or edit one of these
blocks, update the allowlist anchor — and remember: editing the block means
re-verifying the example.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.docs._md_blocks import REPO_ROOT, extract_python_blocks

if TYPE_CHECKING:
    from pathlib import Path

# Self-contained, executable blocks, identified by (markdown path, an anchor
# substring that must appear in the block body). The anchor makes the binding
# robust to small line-number drift and documents *which* block is meant.
EXECUTABLE_BLOCKS: tuple[tuple[str, str], ...] = (
    ("README.md", "async def main() -> None:"),
    ("docs/quickstart.md", 'print("Store ready!")'),
)


def _resolve_block_body(rel_path: str, anchor: str) -> str:
    path = REPO_ROOT / rel_path
    blocks = extract_python_blocks(path)
    matches = [b for b in blocks if anchor in b.body and "asyncio.run(main())" in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one self-contained block in {rel_path} containing "
            f"anchor {anchor!r} and 'asyncio.run(main())', found {len(matches)}. "
            f"Update EXECUTABLE_BLOCKS in {__file__}.",
        )
    return matches[0].body


def _run_snippet(body: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "doc_snippet.py"
    script.write_text(body, encoding="utf-8")
    return subprocess.run(  # noqa: S603 — trusted, repo-authored doc snippet
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    ("rel_path", "anchor"),
    EXECUTABLE_BLOCKS,
    ids=[rel for rel, _ in EXECUTABLE_BLOCKS],
)
def test_self_contained_doc_block_runs(rel_path: str, anchor: str, tmp_path: Path) -> None:
    """A complete, runnable documentation snippet exits 0 against installed engrava."""
    body = _resolve_block_body(rel_path, anchor)
    result = _run_snippet(body, tmp_path)
    assert result.returncode == 0, (
        f"Documentation snippet from {rel_path} exited {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
