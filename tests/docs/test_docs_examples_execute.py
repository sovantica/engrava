"""Layer 1 of the documentation-example tests — execute documentation code.

Compiling a snippet (Layer 2) proves it is *syntactically* valid Python, but it
cannot catch an example that calls an API which does not exist or behaves
differently from what the prose claims (e.g. reading ``result.is_valid`` when
the real attribute is ``result.valid``). This module closes that gap for the
highest-value examples by actually *running* them against the installed
``engrava`` and asserting a clean exit. It offers four execution shapes:

**Self-contained blocks.** Some documentation code blocks are complete, runnable
scripts (they import what they use and drive themselves via ``asyncio.run``).
Each such block is executed exactly as a reader would — written to a temp file
and run in a subprocess — and must exit 0. This is the strongest guarantee: the
published snippet *runs*.

**Self-contained blocks with no async entrypoint.** A block can be equally
self-contained — no undefined collaborator, no disk, no network — while never
awaiting anything at all (a plain construction, a stdlib capability probe, a
logging-config statement). ``EXECUTABLE_BLOCKS`` cannot register one of these:
its own resolver requires the literal ``asyncio.run(main())`` marker, which a
synchronous block never has. ``SYNC_EXECUTABLE_BLOCKS`` below is the same
allowlist-and-run mechanism without that marker requirement, but its guarantee
is narrower, not equal: the async path's marker at least forces the reader to
see the script drive itself, while a synchronous module can exit 0 by merely
defining a function it never calls, with that function's body never running.
``_uncalled_top_level_functions`` rejects that specific shape (a bare
top-level ``def``/``async def`` never invoked in the same module), so what
this path actually proves is: the module executes top to bottom without
raising, and none of its substance hides behind an uncalled top-level
function. It does not trace calls through indirection, and it says nothing
about a method inside a class body that the block never invokes — a class or
``Protocol`` definition sitting uninstantiated is an intended shape for
``DEFINITION_ONLY``, not a gap this check closes.

**Concatenated pages.** Some pages build *one* example across several
*consecutive* code blocks (imports, then a helper, then more helpers, then a
``main()`` that ties them together). No single block runs on its own, but the
contiguous run of blocks concatenated in document order is a complete script.
For an opted-in page this module joins that contiguous run into one script and
runs it in a subprocess, asserting a clean exit — so the whole worked example is
executed against the package, including the return-shape-sensitive search
round-trip in the middle of it.

**Fixture-executed fragments.** A fragment that only *assumes* a store/connection
already exists — the ``ASSUMES_STORE_OR_CONNECTION`` member of
``CompileOnlyReason`` — needs no subprocess and no ``asyncio.run(main())``
wrapper: it runs in-process against a fresh, plain fixture store built exactly
as ``docs/quickstart.md``'s "Create a Store" section builds one. Each such
block executes as written, and each helper it defines is then invoked with
literal arguments **registered per entry** in ``FIXTURE_EXECUTED_BLOCKS`` below
— never guessed from a parameter's name at run time.

Fragment blocks that are neither self-contained, part of an opted-in
concatenated run, nor registered in ``FIXTURE_EXECUTED_BLOCKS`` (they need a
specially-configured store, an on-disk artifact, a live external service, or an
undefined domain value) are out of scope here; they are covered by the compile
+ phantom-API guards in ``test_docs_examples_compile.py`` and by the behaviour
tests in ``test_docs_examples_behavior.py``.

Opting a page in
-----------------
All four execution shapes are **allowlist-driven**: a block runs only when it
has an explicit entry in ``EXECUTABLE_BLOCKS``, ``SYNC_EXECUTABLE_BLOCKS``,
``CONCATENATED_PAGES``, or ``FIXTURE_EXECUTED_BLOCKS`` below. The opt-in
lives entirely in this test module — there is no special fence syntax or
marker in the Markdown — so the public docs (and the engrava.ai mirror) need
no magic annotations to be executed: published Markdown stays clean of any
test-only markers.

* To execute a **single** self-contained block, add a
  ``(markdown_path, anchor_substring)`` entry to ``EXECUTABLE_BLOCKS``. The
  anchor is a short string unique to that block; the block must also drive
  itself via ``asyncio.run(main())``.
* To execute a **single** self-contained block that has no async entrypoint,
  add a ``(markdown_path, anchor_substring)`` entry to
  ``SYNC_EXECUTABLE_BLOCKS`` instead. The anchor must still be unique to that
  block, but the block itself needs no ``asyncio.run(main())`` marker.
* To execute a **contiguous run** of blocks as one page, add a
  ``(markdown_path, first_anchor, last_anchor)`` entry to
  ``CONCATENATED_PAGES``. ``first_anchor`` must appear in exactly one block and
  ``last_anchor`` in exactly one (later or same) block; every block from the
  first match through the last match — inclusive — is concatenated in document
  order. Anchor a contiguous run, **not** a whole page: a page may follow a
  complete example with later illustrative fragments that do not compose, so the
  range is bounded explicitly by its end anchor.
* To execute a **store/connection fragment**, add a
  ``(markdown_path, anchor_substring, invoke)`` entry to
  ``FIXTURE_EXECUTED_BLOCKS``. ``invoke`` is ``None`` when the block needs no
  further call (it already does everything it claims once a store exists), or a
  registered async callback that invokes the block's own helper(s) with literal
  arguments when the block only *defines* one. An ``invoke=None`` entry is
  subject to the same ``_uncalled_top_level_functions`` guard as
  ``SYNC_EXECUTABLE_BLOCKS``: a top-level function it defines but never calls
  fails the test rather than passing on dead code.

When you move or edit one of these blocks, update its anchor — and remember:
editing the block means re-verifying the example.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import os
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import aiosqlite
import pytest

from engrava import (
    LifecycleStatus,
    Priority,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from tests.docs._md_blocks import (
    REPO_ROOT,
    CodeBlock,
    extract_fenced_blocks,
    extract_python_blocks,
    lines_outside_fences,
    markdown_lines,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    # A registered post-execution call for one FIXTURE_EXECUTED_BLOCKS entry: given
    # the block's exec namespace and the fixture's own store, invoke exactly the
    # helper(s) the block defines with literal arguments -- never a parameter-name
    # guess. ``None`` means the block needs no such call (it already runs to
    # completion as a bare fragment).
    _FixtureInvoke = Callable[[dict[str, object], SqliteEngravaCore], Awaitable[None]]

# Bound for every documentation subprocess so a hung example cannot wedge CI.
_RUN_TIMEOUT_S = 120

# Timeout for a fixture-executed fragment (in-process, not a subprocess), applied
# with asyncio.wait_for, which can cancel only at an await -- see
# _run_fixture_block. Shorter than _RUN_TIMEOUT_S because these are fast
# in-memory operations, not a fresh interpreter start.
_FIXTURE_RUN_TIMEOUT_S = 15


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

# Self-contained blocks with no async entrypoint at all -- a plain script that
# imports what it uses and runs top-level, with no store/conn collaborator, no
# disk, and no network. Formerly filed as COMPILE_ONLY with reason
# NO_ASSERTABLE_CLAIM or HARNESS_SHAPE_MISMATCH in test_docs_examples_coverage.py:
# each one runs cleanly, the harness's own asyncio-only requirement was the only
# thing stopping it. Identified by (markdown path, an anchor substring that must
# appear in the block body); unlike EXECUTABLE_BLOCKS, the anchor alone is
# sufficient -- no ``asyncio.run(main())`` marker is required or expected.
SYNC_EXECUTABLE_BLOCKS: tuple[tuple[str, str], ...] = (
    # docs/dreaming.md -- custom DreamingSignalProtocol wiring (sync construction).
    ("docs/dreaming.md", "class MySignal:"),
    # docs/extension-hooks.md -- StructuralSplitProducer FIXED_WINDOW constructor.
    ("docs/extension-hooks.md", 'window_unit="word",'),
    # docs/concepts.md -- ThoughtRecord field tour (construction only).
    ("docs/concepts.md", "inner/outer-speech boundary"),
    # docs/known-limitations.md -- standalone stdlib sqlite3 FTS5 capability probe.
    ("docs/known-limitations.md", "FTS5 is available"),
    # docs/observability.md -- illustrative logging configuration.
    ("docs/observability.md", 'getLogger("engrava").setLevel'),
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


def _uncalled_top_level_functions(body: str) -> list[str]:
    """Return every top-level function ``body`` defines but never calls, sorted.

    A synchronous block can exit 0 while its substance sits inert inside a
    function that is defined but never invoked, e.g.::

        def example():
            StructuralSplitProducer(window_unit="word", nonexistent_argument=True)
        # example() is never called

    ``_run_script`` would report a clean exit for that even though the
    constructor call inside ``example`` never runs. This is a narrow, cheap
    guard, not a coverage tool: it only looks at *top-level* (module-scope)
    ``def``/``async def`` statements and whether their name appears as the
    callee of a ``Call`` node anywhere in the same module. It does not trace
    calls through indirection (a name passed as a callback and invoked later,
    a method reached only via ``getattr``), and it says nothing about a method
    defined inside a class body that the block never invokes -- a class or
    ``Protocol`` definition is expected to sit uninstantiated in exactly that
    way, so this check does not apply to methods, only to bare functions.
    """
    tree = ast.parse(body)
    top_level_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not top_level_names:
        return []
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return sorted(top_level_names - called_names)


def _resolve_sync_block_body(rel_path: str, anchor: str) -> str:
    """Resolve a ``SYNC_EXECUTABLE_BLOCKS`` entry, with no async-marker requirement.

    Mirrors ``_resolve_block_body`` except it does not filter on
    ``"asyncio.run(main())"`` -- the whole point of this registry is a block
    that never awaits anything, so requiring that marker would defeat it. In
    its place, this rejects a block whose only top-level function is never
    called (see ``_uncalled_top_level_functions``): without that guard, exit-0
    would be satisfied by defining dead code, not by running it.
    """
    path = REPO_ROOT / rel_path
    blocks = extract_python_blocks(path)
    matches = [b for b in blocks if anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one block in {rel_path} containing anchor "
            f"{anchor!r}, found {len(matches)}. Update SYNC_EXECUTABLE_BLOCKS "
            f"in {__file__}.",
        )
    body = matches[0].body
    uncalled = _uncalled_top_level_functions(body)
    if uncalled:
        pytest.fail(
            f"{rel_path}: the block anchored on {anchor!r} defines top-level "
            f"function(s) {uncalled} that are never called. Exiting 0 would prove "
            f"only that the module compiles and its top-level statements ran, not "
            f"that this function's body ever executes. Either call it within the "
            f"block, or this block does not belong in SYNC_EXECUTABLE_BLOCKS.",
        )
    return body


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
    ("rel_path", "anchor"),
    SYNC_EXECUTABLE_BLOCKS,
    ids=[rel for rel, _ in SYNC_EXECUTABLE_BLOCKS],
)
def test_sync_self_contained_doc_block_runs(rel_path: str, anchor: str, tmp_path: Path) -> None:
    """A self-contained snippet with no async entrypoint runs top to bottom without raising.

    Closes the gap ``HARNESS_SHAPE_MISMATCH`` names: ``EXECUTABLE_BLOCKS`` only
    matches a block containing the literal ``asyncio.run(main())`` marker, so a
    synchronous self-contained script -- no store/conn, no disk, no network --
    could never be executed even though nothing about the example itself
    prevents running it. ``_resolve_sync_block_body`` also rejects a block whose
    only top-level function is never called, so a clean exit here is not
    satisfied by merely defining dead code (see ``_uncalled_top_level_functions``).
    That guard is narrower than proof every line ran: it does not see a method
    inside a class body that the block never invokes.
    """
    body = _resolve_sync_block_body(rel_path, anchor)
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
    lines = lines_outside_fences(
        markdown_lines((REPO_ROOT / rel_path).read_text(encoding="utf-8")), rel_path
    )
    paragraphs: list[str] = []
    current: list[str] = []
    for raw in lines:
        if raw.strip(" \t"):
            current.append(raw)
        elif current:
            paragraphs.append(" ".join(" ".join(current).split()))
            current = []
    if current:
        paragraphs.append(" ".join(" ".join(current).split()))
    return paragraphs


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        pytest.param(
            "Before.\n\n```text\nA hidden claim.\n```\n\nAfter.\n",
            ["Before.", "After."],
            id="fence-hides-its-text",
        ),
        pytest.param(
            "Before.\n\n    ```\n\nA claim.\n\n    ```\n\nAfter.\n",
            ["Before.", "```", "A claim.", "```", "After."],
            id="four-spaces-is-not-a-fence",
        ),
        pytest.param(
            "Before.\n\n```text\n```\u00a0\nA hidden claim.\n```\n\nAfter.\n",
            ["Before.", "After."],
            id="no-break-space-does-not-close",
        ),
        pytest.param(
            "First line\nsecond line\n\nNext.\n",
            ["First line second line", "Next."],
            id="paragraphs-split-on-blank-lines",
        ),
    ],
)
def test_page_prose_paragraphs_drops_exactly_the_fenced_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    page: str,
    expected: list[str],
) -> None:
    """The paragraphs are the page's text outside real fences, one per blank-line run.

    The fence rule is the one every documentation gate shares: four spaces open
    no fence, and a no-break space after the backticks does not close one.
    """
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    (tmp_path / "page.md").write_text(page, encoding="utf-8")

    assert _page_prose_paragraphs("page.md") == expected


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


# ---------------------------------------------------------------------------
# Fixture-executed fragments — a fragment that only assumes an existing
# store/connection, run in-process against a fresh fixture store.
# ---------------------------------------------------------------------------


async def _fresh_fixture_store(conn: aiosqlite.Connection) -> SqliteEngravaCore:
    """Build the plain store every ``ASSUMES_STORE_OR_CONNECTION`` block is promised.

    Mirrors ``docs/quickstart.md``'s "Create a Store" section exactly: a bare
    ``SqliteEngravaCore(conn)`` over an aiosqlite connection with
    ``row_factory = aiosqlite.Row``, schema applied via ``ensure_schema()`` --
    no extra constructor keyword, no wrapper class, no on-disk file. A block
    needing more than this belongs to a different ``CompileOnlyReason`` member
    (``REQUIRES_SPECIALLY_CONFIGURED_STORE``, ``REQUIRES_ON_DISK_ARTIFACT``, ...),
    not this one.
    """
    conn.row_factory = aiosqlite.Row
    store = SqliteEngravaCore(conn)
    await store.ensure_schema()
    return store


async def _invoke_verify_journal_with_lock_retry(
    ns: dict[str, object], store: SqliteEngravaCore
) -> None:
    """docs/error-handling.md — call with only the store; ``attempts`` keeps its default."""
    fn = cast(
        "Callable[[SqliteEngravaCore], Awaitable[object]]", ns["verify_journal_with_lock_retry"]
    )
    await fn(store)


async def _invoke_recall_with_degradation_flags(
    ns: dict[str, object], store: SqliteEngravaCore
) -> None:
    """docs/error-handling.md — call with a literal query string."""
    fn = cast(
        "Callable[[SqliteEngravaCore, str], Awaitable[object]]",
        ns["recall_with_degradation_flags"],
    )
    await fn(store, "what does the user prefer?")


async def _invoke_journal_ok(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/observability.md — call with only the store."""
    fn = cast("Callable[[SqliteEngravaCore], Awaitable[object]]", ns["journal_ok"])
    await fn(store)


async def _invoke_healthcheck(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/observability.md — call with only the store."""
    fn = cast("Callable[[SqliteEngravaCore], Awaitable[object]]", ns["healthcheck"])
    await fn(store)


async def _invoke_store_percept(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/guides/agent-memory.md — call with literal text/cycle/user/session/turn."""
    fn = cast(
        "Callable[[SqliteEngravaCore, str, int, str, str, int], Awaitable[object]]",
        ns["store_percept"],
    )
    await fn(store, "The user prefers dark mode.", 1, "user-1", "session-1", 0)


async def _invoke_store_turn(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/recipes/index.md — call with literal turn text and conversation metadata."""
    fn = cast(
        "Callable[..., Awaitable[object]]",
        ns["store_turn"],
    )
    await fn(
        store,
        "What's the weather like?",
        "It's sunny today.",
        cycle=1,
        session_id="session-1",
        turn_index=0,
        user_id="user-1",
    )


async def _invoke_context_for(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/recipes/index.md — call with a literal query and cycle."""
    fn = cast(
        "Callable[[SqliteEngravaCore, str, int], Awaitable[object]]",
        ns["context_for"],
    )
    await fn(store, "weather", 1)


async def _invoke_search_in_session(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/recipes/index.md — call with a literal query, session id, and cycle."""
    fn = cast(
        "Callable[[SqliteEngravaCore, str, str, int], Awaitable[object]]",
        ns["search_in_session"],
    )
    await fn(store, "weather", "session-1", 1)


async def _invoke_assemble_unit(ns: dict[str, object], store: SqliteEngravaCore) -> None:
    """docs/search.md — seed one chunk the block's own filters can find, then call it.

    ``assemble_unit`` never creates the unit it reads (the page's prose calls it a
    caller-side recipe over data written earlier), so this seeds a single thought
    with the ``session_id``/``turn_index``/``chunk_index`` metadata the block's own
    ``FieldPredicate`` filters key on, mirroring the shape
    ``docs/recipes/index.md``'s ``store_turn`` helper writes -- a plain
    ``ThoughtRecord`` with conversation-scoping metadata, nothing the block would
    not otherwise assume already exists.
    """
    seed = ThoughtRecord(
        thought_id="seed-chunk-0",
        thought_type=ThoughtType.OBSERVATION,
        essence="What's the weather like?",
        content="What's the weather like?",
        priority=Priority.P2,
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_cycle=0,
        updated_cycle=0,
        source="user-1",
        metadata={"session_id": "session-1", "turn_index": 0, "chunk_index": 0},
    )
    await store.create_thought(seed)
    fn = cast("Callable[[str, str], Awaitable[object]]", ns["assemble_unit"])
    await fn("weather", "seed-chunk-0")


# Fragments that only assume an existing store/connection -- the
# ``ASSUMES_STORE_OR_CONNECTION`` member of ``CompileOnlyReason`` -- identified by
# (markdown path, anchor substring, invoke). ``invoke`` is ``None`` when the block
# already does everything it claims once a store exists; otherwise it is a
# registered callback (above) that calls the block's own helper with literal
# arguments. This is the source of truth these 35 promoted blocks live in;
# ``test_docs_examples_coverage.py`` folds their locations into the executed side.
FIXTURE_EXECUTED_BLOCKS: tuple[tuple[str, str, _FixtureInvoke | None], ...] = (
    (
        "docs/api-reference.md",
        'FieldPredicate("$.subtype", FieldOp.EQ, "supports")',
        None,
    ),
    (
        "docs/error-handling.md",
        "async def verify_journal_with_lock_retry",
        _invoke_verify_journal_with_lock_retry,
    ),
    (
        "docs/error-handling.md",
        "async def recall_with_degradation_flags",
        _invoke_recall_with_degradation_flags,
    ),
    ("docs/cli.md", "print(result.valid)", None),
    ("docs/concepts.md", "cycle_provider=StaticCycleProvider(0)", None),
    ("docs/concepts.md", "resume_from = await store.max_cycle()", None),
    ("docs/dreaming.md", "graph_edge_decay=0.3", None),
    ("docs/extensions.md", "discover_manifests()", None),
    (
        "docs/guides/agent-memory.md",
        "async def store_percept",
        _invoke_store_percept,
    ),
    (
        "docs/guides/agent-memory.md",
        "cycle = await store.max_cycle()   # the highest cycle stored",
        None,
    ),
    ("docs/guides/migrating-from-other-memory.md", "memory.add(", None),
    ("docs/guides/migrating-from-other-memory.md", "hits = memory.search(", None),
    (
        "docs/guides/migrating-from-other-memory.md",
        "over-fetch, then filter and trim",
        None,
    ),
    ("docs/guides/migrating-from-other-memory.md", "json_extract(metadata_json", None),
    (
        "docs/guides/migrating-from-other-memory.md",
        'allowed={"public"}, owner="u1"',
        None,
    ),
    ("docs/mindql.md", "Active thoughts: {count_result.count}", None),
    (
        "docs/mindql.md",
        "from engrava import parse\n\nresult = await store.execute_mindql",
        None,
    ),
    ("docs/observability.md", "async def journal_ok", _invoke_journal_ok),
    ("docs/observability.md", "async def healthcheck", _invoke_healthcheck),
    ("docs/quickstart.md", 'recall("what does the user prefer?")', None),
    ("docs/quickstart.md", "Python's async ecosystem and rich ML libraries", None),
    ("docs/quickstart.md", "returns (thought_id, bm25_score) tuples", None),
    ("docs/quickstart.md", "Found {len(result.rows)} thoughts", None),
    ("docs/recipes/index.md", "async def store_turn", _invoke_store_turn),
    ("docs/recipes/index.md", "async def context_for", _invoke_context_for),
    ("docs/recipes/index.md", "async def search_in_session", _invoke_search_in_session),
    ("docs/recipes/index.md", "resume from the stored high-water mark", None),
    ("docs/search.md", "print(store.fts_match_failure_count)", None),
    ("docs/search.md", "the caller owns", None),
    ("docs/search.md", "OR-matched", None),
    ("docs/search.md", "python async", None),
    ("docs/search.md", "async def assemble_unit", _invoke_assemble_unit),
    ("docs/troubleshooting.md", "['fts5', 'priority', 'recency']", None),
    ("docs/troubleshooting.md", "require an exact phrase", None),
    ("docs/troubleshooting.md", "lower it if nothing clears the bar", None),
)


def _resolve_fixture_block_body(rel_path: str, anchor: str, invoke: _FixtureInvoke | None) -> str:
    """Resolve a ``FIXTURE_EXECUTED_BLOCKS`` entry, guarding the ``invoke=None`` shape.

    When ``invoke`` is ``None`` the block is expected to do its work as written,
    with no external call into a helper it defines -- exactly the shape
    ``_resolve_sync_block_body`` guards for ``SYNC_EXECUTABLE_BLOCKS``, and for
    the same reason: a block could define a top-level function and never call
    it, in which case exiting 0 would prove only that the module compiles, not
    that the function's body ever ran. When ``invoke`` is a registered
    callback, an uncalled top-level function is the intended shape -- the
    callback is what invokes the block's own helper -- so the guard does not
    apply.
    """
    path = REPO_ROOT / rel_path
    matches = [b for b in extract_python_blocks(path) if anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"Expected exactly one block in {rel_path} containing anchor {anchor!r}, "
            f"found {len(matches)}. Update FIXTURE_EXECUTED_BLOCKS in {__file__}.",
        )
    body = matches[0].body
    if invoke is None:
        uncalled = _uncalled_top_level_functions(body)
        if uncalled:
            pytest.fail(
                f"{rel_path}: the block anchored on {anchor!r} defines top-level "
                f"function(s) {uncalled} that are never called and has no registered "
                f"invoke callback. Exiting 0 would prove only that the block's other "
                f"top-level statements ran, not that this function's body ever "
                f"executes. Either call it within the block, or register an invoke "
                f"callback in FIXTURE_EXECUTED_BLOCKS that calls it.",
            )
    return body


async def _run_fixture_block(body: str, invoke: _FixtureInvoke | None) -> None:
    """Execute one ``ASSUMES_STORE_OR_CONNECTION`` fragment against a fresh fixture.

    Per-block isolation: a brand-new in-memory connection and store are built for
    every call, so one block's rows are never what the next block's query finds.
    Only a block that contains a top-level ``await``, and the registered
    ``invoke`` call, run under ``asyncio.wait_for``. A block without one runs
    inside ``eval()`` with no timeout, and ``wait_for`` can cancel only at an
    ``await``, so code that never yields is not bounded.
    """
    conn = await aiosqlite.connect(":memory:")
    try:
        store = await _fresh_fixture_store(conn)
        ns: dict[str, object] = {
            "store": store,
            "conn": conn,
            "db": conn,
            "__name__": "__doc_fixture__",
        }
        code = compile(body, "<doc-fixture-block>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        result = eval(code, ns)  # noqa: S307 - trusted, repo-authored doc snippet
        if inspect.iscoroutine(result):
            await asyncio.wait_for(result, _FIXTURE_RUN_TIMEOUT_S)
        if invoke is not None:
            await asyncio.wait_for(invoke(ns, store), _FIXTURE_RUN_TIMEOUT_S)
    finally:
        await conn.close()


@pytest.mark.parametrize(
    ("rel_path", "anchor", "invoke"),
    FIXTURE_EXECUTED_BLOCKS,
    ids=[f"{rel}#{i}" for i, (rel, _anchor, _invoke) in enumerate(FIXTURE_EXECUTED_BLOCKS)],
)
async def test_fixture_executed_doc_block_runs(
    rel_path: str,
    anchor: str,
    invoke: _FixtureInvoke | None,
) -> None:
    """A fragment that only assumes a store/connection runs against a fresh fixture.

    Promotes the 35 ``ASSUMES_STORE_OR_CONNECTION`` compile-only blocks out of
    the compile-only tier: each executes against exactly the
    plain store ``docs/quickstart.md``'s "Create a Store" section builds, and each
    helper it defines is invoked with literal arguments registered in
    ``FIXTURE_EXECUTED_BLOCKS`` above -- never guessed from a parameter's name.
    """
    body = _resolve_fixture_block_body(rel_path, anchor, invoke)
    await _run_fixture_block(body, invoke)
