"""Layer 1 of the documentation-example tests — execute documentation code.

Compiling a snippet (Layer 2) proves it is *syntactically* valid Python, but it
cannot catch an example that calls an API which does not exist or behaves
differently from what the prose claims (e.g. reading ``result.is_valid`` when
the real attribute is ``result.valid``). This module closes that gap for the
highest-value examples by actually *running* them against the installed
``engrava`` and asserting a clean exit. It offers two execution shapes:

**Self-contained blocks.** Some documentation code blocks are complete, runnable
scripts (they import what they use and drive themselves via ``asyncio.run``).
Each such block is executed exactly as a reader would — written to a temp file
and run in a subprocess — and must exit 0. This is the strongest guarantee: the
published snippet *runs*.

**Concatenated pages.** Some pages build *one* example across several
*consecutive* code blocks (imports, then a helper, then more helpers, then a
``main()`` that ties them together). No single block runs on its own, but the
contiguous run of blocks concatenated in document order is a complete script.
For an opted-in page this module joins that contiguous run into one script and
runs it in a subprocess, asserting a clean exit — so the whole worked example is
executed against the package, including the return-shape-sensitive search
round-trip in the middle of it.

Fragment blocks that are neither self-contained nor part of an opted-in
concatenated run (the majority — they assume an existing ``store``/``conn`` or
show a class definition) are out of scope here; they are covered by the
compile + phantom-API guards in ``test_docs_examples_compile.py`` and by the
behaviour tests in ``test_docs_examples_behavior.py``.

Opting a page in
-----------------
Both execution shapes are **allowlist-driven**: a page runs only when it has an
explicit entry in ``EXECUTABLE_BLOCKS`` or ``CONCATENATED_PAGES`` below. The
opt-in lives entirely in this test module — there is no special fence syntax or
marker in the Markdown — so the public docs (and the engrava.ai mirror) need no
magic annotations to be executed (cf. AGENT_PRINCIPLES Principle 1).

* To execute a **single** self-contained block, add a
  ``(markdown_path, anchor_substring)`` entry to ``EXECUTABLE_BLOCKS``. The
  anchor is a short string unique to that block; the block must also drive
  itself via ``asyncio.run(main())``.
* To execute a **contiguous run** of blocks as one page, add a
  ``(markdown_path, first_anchor, last_anchor)`` entry to
  ``CONCATENATED_PAGES``. ``first_anchor`` must appear in exactly one block and
  ``last_anchor`` in exactly one (later or same) block; every block from the
  first match through the last match — inclusive — is concatenated in document
  order. Anchor a contiguous run, **not** a whole page: a page may follow a
  complete example with later illustrative fragments that do not compose, so the
  range is bounded explicitly by its end anchor.

When you move or edit one of these blocks, update its anchor — and remember:
editing the block means re-verifying the example.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.docs._md_blocks import REPO_ROOT, CodeBlock, extract_python_blocks

if TYPE_CHECKING:
    from pathlib import Path

# Bound for every documentation subprocess so a hung example cannot wedge CI.
_RUN_TIMEOUT_S = 120

# Self-contained, executable blocks, identified by (markdown path, an anchor
# substring that must appear in the block body). The anchor makes the binding
# robust to small line-number drift and documents *which* block is meant.
EXECUTABLE_BLOCKS: tuple[tuple[str, str], ...] = (
    ("README.md", "async def main() -> None:"),
    ("docs/quickstart.md", 'print("Store ready!")'),
    ("docs/guides/migrating-from-other-memory.md", "Imported {total} thoughts."),
)

# Pages that build one example across a contiguous run of code blocks, identified
# by (markdown path, first-block anchor, last-block anchor). The two anchors bound
# an inclusive, contiguous range of blocks that is concatenated in document order
# and run as a single script. Anchor the runnable run, not the whole page.
CONCATENATED_PAGES: tuple[tuple[str, str, str], ...] = (
    # docs/tutorial.md builds one notes-memory example across five consecutive
    # blocks: imports + embed() -> NOTES + ingest() -> link() -> search()
    # (the search_hybrid round-trip) -> main() + asyncio.run. The whole run
    # composes into a complete script; there is no trailing non-composing block.
    (
        "docs/tutorial.md",
        "def embed(text: str) -> list[float]:",
        "asyncio.run(main())",
    ),
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


def _unique_block_index(blocks: list[CodeBlock], rel_path: str, anchor: str, role: str) -> int:
    matches = [i for i, b in enumerate(blocks) if anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one block in {rel_path} containing the {role} anchor "
            f"{anchor!r}, found {len(matches)}. Update CONCATENATED_PAGES in {__file__}.",
        )
    return matches[0]


def _resolve_page_script(rel_path: str, first_anchor: str, last_anchor: str) -> str:
    """Concatenate the inclusive, contiguous block range bounded by the anchors."""
    path = REPO_ROOT / rel_path
    blocks = extract_python_blocks(path)
    start = _unique_block_index(blocks, rel_path, first_anchor, "first")
    end = _unique_block_index(blocks, rel_path, last_anchor, "last")
    if end < start:
        pytest.fail(
            f"In {rel_path} the last anchor {last_anchor!r} (block {end}) precedes the "
            f"first anchor {first_anchor!r} (block {start}). Update CONCATENATED_PAGES "
            f"in {__file__}.",
        )
    return "\n\n".join(b.body for b in blocks[start : end + 1])


def _run_script(body: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "doc_snippet.py"
    script.write_text(body, encoding="utf-8")
    return subprocess.run(  # noqa: S603 — trusted, repo-authored doc snippet
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_S,
    )


@pytest.mark.parametrize(
    ("rel_path", "anchor"),
    EXECUTABLE_BLOCKS,
    ids=[rel for rel, _ in EXECUTABLE_BLOCKS],
)
def test_self_contained_doc_block_runs(rel_path: str, anchor: str, tmp_path: Path) -> None:
    """A complete, runnable documentation snippet exits 0 against installed engrava."""
    body = _resolve_block_body(rel_path, anchor)
    result = _run_script(body, tmp_path)
    assert result.returncode == 0, (
        f"Documentation snippet from {rel_path} exited {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.mark.parametrize(
    ("rel_path", "first_anchor", "last_anchor"),
    CONCATENATED_PAGES,
    ids=[rel for rel, _, _ in CONCATENATED_PAGES],
)
def test_concatenated_doc_page_runs(
    rel_path: str,
    first_anchor: str,
    last_anchor: str,
    tmp_path: Path,
) -> None:
    """A page's contiguous run of blocks, concatenated, exits 0 against installed engrava.

    This executes a worked example that is split across several consecutive doc
    blocks and is therefore not runnable as any single block — catching API
    drift in the mid-example fragments (e.g. the search round-trip) that
    compile-only checks cannot see.
    """
    script = _resolve_page_script(rel_path, first_anchor, last_anchor)
    result = _run_script(script, tmp_path)
    assert result.returncode == 0, (
        f"Concatenated documentation page {rel_path} exited {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
