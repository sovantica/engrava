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
magic annotations to be executed: published Markdown stays clean of any
test-only markers.

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

import ast
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.docs._md_blocks import (
    REPO_ROOT,
    CodeBlock,
    extract_fenced_blocks,
    extract_python_blocks,
)

if TYPE_CHECKING:
    from pathlib import Path

# Bound for every documentation subprocess so a hung example cannot wedge CI.
_RUN_TIMEOUT_S = 120


def _isolated_child_env() -> dict[str, str]:
    """Return a deterministic, offline, single-threaded environment for a snippet.

    Documentation snippets are run in a fresh subprocess. Forcing the offline
    flags makes the run network-independent regardless of the caller's ambient
    environment, and pinning the native thread pools keeps a snippet that
    happens to import a heavy dependency from contending for native resources
    with the rest of the suite.

    Returns:
        A copy of ``os.environ`` with the deterministic overrides applied.

    """
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


# Self-contained, executable blocks, identified by (markdown path, an anchor
# substring that must appear in the block body). The anchor makes the binding
# robust to small line-number drift and documents *which* block is meant.
EXECUTABLE_BLOCKS: tuple[tuple[str, str], ...] = (
    ("README.md", "async def main() -> None:"),
    ("docs/quickstart.md", 'print("Store ready!")'),
    ("docs/guides/migrating-from-other-memory.md", "Imported {total} thoughts."),
    # docs/bitemporal.md — three self-contained valid-time examples.
    ("docs/bitemporal.md", "# valid_until omitted -> open upper bound -> still valid"),
    ("docs/bitemporal.md", "assert len(march.rows) == 1  # inside the valid window"),
    ("docs/bitemporal.md", "await store.invalidate_thought("),
    # docs/evidence-and-conflicts.md — the end-to-end single-value-slot conflict
    # workflow: create evidence + claims, detect the conflict with a caller-owned
    # rule, record a CONTESTED_BY edge, and open a clarification task.
    ("docs/evidence-and-conflicts.md", "incompatible_single_value_claims"),
)

# docs/tutorial.md builds one notes-memory example across five consecutive
# blocks: imports + embed() -> NOTES + ingest() -> link() -> search()
# (the search_hybrid round-trip) -> main() + asyncio.run. The whole run
# composes into a complete script; there is no trailing non-composing block.
_TUTORIAL_PAGE: tuple[str, str, str] = (
    "docs/tutorial.md",
    "def embed(text: str) -> list[float]:",
    "asyncio.run(main())",
)

# Pages that build one example across a contiguous run of code blocks, identified
# by (markdown path, first-block anchor, last-block anchor). The two anchors bound
# an inclusive, contiguous range of blocks that is concatenated in document order
# and run as a single script. Anchor the runnable run, not the whole page.
CONCATENATED_PAGES: tuple[tuple[str, str, str], ...] = (_TUTORIAL_PAGE,)

# docs/tutorial.md publishes the output of its own example as a ``text`` block —
# the ranked notes, their scores, the signal list, and the stored count — and
# then reasons about that output in prose. Executing the page proves only that
# it exits 0; the transcript is where the page makes a claim a reader will check.
# Anchor it here and compare it against a real run, so the two cannot diverge.
_TUTORIAL_TRANSCRIPT_ANCHOR = "Query: 'anything about coffee?'"

# The tutorial's ``NOTES`` list, so the test can work out which notes the example
# ranked and which it did not without being told either set.
_TUTORIAL_NOTES_ANCHOR = "NOTES = ["

# The page's exclusion claim. The paragraph carrying this phrase must name every
# note the example drops and no note it ranks — both directions derived from the
# run, never written down here.
_TUTORIAL_EXCLUSION_PHRASE = "never reaches `top_k=3`"


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
        # The snippets read nothing from stdin; closing it removes a stdin-
        # inheritance wedge when the parent runs under pytest's output capture.
        stdin=subprocess.DEVNULL,
        env=_isolated_child_env(),
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


def _documented_transcript(rel_path: str, anchor: str) -> str:
    """Return the ``text`` block a page publishes as its example's output."""
    path = REPO_ROOT / rel_path
    matches = [b for b in extract_fenced_blocks(path, "text") if anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one ```text block in {rel_path} containing "
            f"{anchor!r}, found {len(matches)}. The page must publish the output "
            f"its prose reasons about, so this test can compare the two.",
        )
    return matches[0].body


def _tutorial_notes(rel_path: str) -> list[str]:
    """Return the string literals of the page's own ``NOTES`` list."""
    path = REPO_ROOT / rel_path
    matches = [b for b in extract_python_blocks(path) if _TUTORIAL_NOTES_ANCHOR in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one block in {rel_path} containing "
            f"{_TUTORIAL_NOTES_ANCHOR!r}, found {len(matches)}.",
        )
    module = ast.parse(matches[0].body)
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "NOTES" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return [ast.literal_eval(element) for element in node.value.elts]
    pytest.fail(f"{rel_path} has no literal NOTES list to read.")


def _page_prose_paragraphs(rel_path: str) -> list[str]:
    """Return a page's prose paragraphs, fenced blocks removed.

    A note's text appears inside the example that defines it and inside the
    transcript that ranks it, so "the page says something about this note" is
    only evidence of a *claim* once the fenced blocks are stripped out. Each
    paragraph has its whitespace collapsed, so a quoted sentence still matches
    when Markdown wraps it across lines. Splitting on blank lines keeps the unit
    small enough that "this paragraph claims X about this note" means something:
    a page-wide search would let a claim about one note be satisfied by a
    sentence about another.
    """
    lines = (REPO_ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for raw in lines:
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if raw.strip():
            current.append(raw)
        elif current:
            paragraphs.append(" ".join(" ".join(current).split()))
            current = []
    if current:
        paragraphs.append(" ".join(" ".join(current).split()))
    return paragraphs


def test_tutorial_page_produces_the_output_it_publishes(tmp_path: Path) -> None:
    """docs/tutorial.md's published transcript equals what running the page prints.

    The tutorial states which notes rank for its coffee query and reasons about
    the order in prose. Running the page proves only that it exits 0, so a page
    promising a ranking its own toy embedding cannot produce stays green in that
    tier — which is how a plural "the coffee notes rank" survived here.

    The expectation is therefore not written in this file: it is read out of the
    ``text`` block the page publishes as its own output. A page that documents a
    result it does not produce fails here, and a page whose prose reasons about
    numbers the run does not produce fails with it, because the block carries the
    scores and the signal list too.
    """
    rel_path, first_anchor, last_anchor = _TUTORIAL_PAGE
    documented = _documented_transcript(rel_path, _TUTORIAL_TRANSCRIPT_ANCHOR)

    script = _resolve_page_script(rel_path, first_anchor, last_anchor)
    result = _run_script(script, tmp_path)
    # Precondition, not a self-report: without a clean exit there is no output to
    # compare and the assertion below would report a misleading difference.
    assert result.returncode == 0, (
        f"Documentation page {rel_path} exited {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    assert result.stdout.strip() == documented.strip(), (
        f"{rel_path} publishes an output block its own example does not produce. "
        f"Re-read the page's prose against the real output before changing "
        f"either.\n--- documented ---\n{documented}\n--- actual ---\n{result.stdout}"
    )


def test_tutorial_page_claims_exclusion_for_exactly_the_notes_it_drops() -> None:
    """docs/tutorial.md's exclusion claim matches the notes its example drops.

    The page's own example defines four notes and ranks three. Which three is a
    property of the run, so this test derives both sets rather than being told
    either, then reads the page's exclusion claim — the prose paragraph carrying
    ``never reaches `top_k=3``` — and requires it to name every dropped note and
    no ranked one.

    Both directions matter and each catches a different regression. Dropping the
    claim, or going back to "the coffee notes rank for the coffee query" while
    one of them does not, fails the first. Moving the claim onto a note that in
    fact ranks fails the second. Between them, the page cannot go quiet about a
    note it drops, and cannot attribute the exclusion to the wrong note.

    The unit is the paragraph, not the page: a page-wide search would let a
    sentence about one note satisfy the claim owed to another. A dropped note is
    then confined to that paragraph, so a second paragraph elsewhere cannot make
    a competing claim about it.

    What this still does not do is parse English. It pins which notes the page's
    ranking claims are about; it cannot detect an arbitrary contradictory
    sentence that names no note at all. The sibling test above is what pins the
    numbers.
    """
    rel_path, _, _ = _TUTORIAL_PAGE
    notes = _tutorial_notes(rel_path)
    documented = _documented_transcript(rel_path, _TUTORIAL_TRANSCRIPT_ANCHOR)

    excluded = [note for note in notes if note not in documented]
    ranked = [note for note in notes if note in documented]
    # Preconditions: the example must both drop something and rank something, or
    # one of the two directions below would pass vacuously.
    assert excluded, (
        f"{rel_path} ranks every note it defines, so this test proves nothing. "
        f"Either the example changed or the transcript is stale."
    )
    assert ranked, f"{rel_path} ranks none of its notes; the transcript is stale."

    claims = [p for p in _page_prose_paragraphs(rel_path) if _TUTORIAL_EXCLUSION_PHRASE in p]
    assert len(claims) == 1, (
        f"Expected exactly one prose paragraph in {rel_path} containing "
        f"{_TUTORIAL_EXCLUSION_PHRASE!r}, found {len(claims)}. The page must state "
        f"once, in prose, which note its example leaves out."
    )
    claim = claims[0]

    for note in excluded:
        assert note in claim, (
            f"{rel_path} does not rank {note!r} — it is absent from the output the "
            f"page publishes — but the page's exclusion sentence does not name it. "
            f"A reader is left believing every note ranks.\n{claim}"
        )
    others = [p for p in _page_prose_paragraphs(rel_path) if _TUTORIAL_EXCLUSION_PHRASE not in p]
    for note in excluded:
        for paragraph in others:
            assert note not in paragraph, (
                f"{rel_path} discusses {note!r} outside its exclusion sentence. The "
                f"example does not rank that note, so a second paragraph about it "
                f"can only compete with the claim that it is left out.\n{paragraph}"
            )
    for note in ranked:
        assert note not in claim, (
            f"{rel_path} says {note!r} is left out of top_k=3, but the output the "
            f"page publishes ranks it.\n{claim}"
        )
