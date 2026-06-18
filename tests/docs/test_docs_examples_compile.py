"""Layer 2 of the documentation-example tests — static guards on every block.

This module is the broad safety net: it covers **every** fenced ``python``
code block across ``README.md`` and ``docs/`` (including illustrative
fragments that cannot be executed standalone). Two cheap, fragment-safe
checks run on each block:

1. **It compiles.** ``compile()`` catches ``SyntaxError`` without executing
   anything — so even a bare-fragment snippet is verified to be syntactically
   valid Python.
2. **It names no phantom API.** A small denylist of tokens that are known
   *not* to exist in engrava (fabricated enum members, removed/renamed
   symbols, wrong attribute names) must never appear in a doc code block.
   This is the inverse of a secret-scanner: it enumerates only **public,
   nonexistent** identifiers, so it is safe to ship and cannot leak any
   internal term.

These guards are what would have caught the 0.3.x documentation drift
(``thought_type="INSIGHT"``, ``create_edge(..., "ASSOCIATION")``,
``config.db_path``, ``result.row_count``, ...) automatically.

.. important::
   If you change any documentation code block, re-run this suite. If you add
   a genuinely new, legitimate identifier that trips the denylist, update the
   denylist deliberately — do not loosen the pattern to make a real drift
   pass.
"""

from __future__ import annotations

import ast

import pytest

from tests.docs._md_blocks import CodeBlock, all_python_blocks

# Documentation fragments legitimately use top-level ``await`` (they are shown
# as snippets meant to live inside an ``async`` context). Compile with the
# top-level-await flag — the same allowance the async REPL and notebooks use —
# so a real ``SyntaxError`` still fails while a bare ``await store...`` line
# does not.
_COMPILE_FLAGS = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

# Tokens that are known NOT to exist in the engrava public API. Any appearance
# in a documentation code block is a drift bug. Each entry is paired with the
# real replacement so the failure message is actionable. These are all public,
# nonexistent identifiers — never internal/secret terms.
PHANTOM_TOKENS: dict[str, str] = {
    # Fabricated ThoughtType members (real: TASK/OBSERVATION/BELIEF/REFLECTION/OUTPUT_DRAFT/NOTE)
    '"INSIGHT"': "a real ThoughtType, e.g. BELIEF",
    "'INSIGHT'": "a real ThoughtType, e.g. BELIEF",
    "=INSIGHT": "a real ThoughtType, e.g. =OBSERVATION",
    # Fabricated EdgeType members (real: ASSOCIATED, CONSOLIDATED_FROM, ...)
    '"ASSOCIATION"': "EdgeType.ASSOCIATED",
    "'ASSOCIATION'": "EdgeType.ASSOCIATED",
    # Wrong constructor / attribute names
    "config.db_path": "config.database_path",
    "db_path=": "an aiosqlite connection (SqliteEngravaCore(conn)) or from_config",
    ".row_count": ".count (MindQLResult.count)",
    "help_text=": "description= (MindQLExtension.description)",
    # Removed / nonexistent methods
    ".consolidate(": ".run_consolidation(",
    # Fabricated DreamingGates fields
    "min_confidence=": "min_confirmations= / promote_threshold=",
    "min_confirmation_count=": "min_confirmations=",
    "min_composite_score=": "promote_threshold= (on DreamingConfig)",
    # Wrong MindQL parser result type
    "cmd.verb": "query.command (parse returns a MindQLQuery)",
    "cmd.filters": "query.conditions",
}

_ALL_BLOCKS = all_python_blocks()


def _block_id(block: CodeBlock) -> str:
    return block.location


@pytest.mark.parametrize("block", _ALL_BLOCKS, ids=[_block_id(b) for b in _ALL_BLOCKS])
def test_doc_block_compiles(block: CodeBlock) -> None:
    """Every documentation ``python`` block is syntactically valid Python."""
    try:
        compile(block.body, filename=block.location, mode="exec", flags=_COMPILE_FLAGS)
    except SyntaxError as exc:  # pragma: no cover - failure path asserted below
        pytest.fail(
            f"Documentation code block at {block.location} does not compile: "
            f"{exc.msg} (line {exc.lineno}). Fix the snippet in the source "
            f"Markdown file.",
        )


@pytest.mark.parametrize("block", _ALL_BLOCKS, ids=[_block_id(b) for b in _ALL_BLOCKS])
def test_doc_block_has_no_phantom_api(block: CodeBlock) -> None:
    """No documentation block references a known-nonexistent engrava identifier."""
    for token, replacement in PHANTOM_TOKENS.items():
        assert token not in block.body, (
            f"Documentation block at {block.location} references {token!r}, "
            f"which is not part of the engrava public API. Use {replacement} "
            f"instead, and re-verify the example against the shipped code."
        )


def test_documentation_has_python_examples() -> None:
    """Sanity check: the extractor found documentation blocks to guard.

    Guards against a silent regression where a path change makes
    ``all_python_blocks()`` return nothing and the parametrized tests above
    vacuously pass.
    """
    assert len(_ALL_BLOCKS) > 20, (
        f"expected the docs to contain many python blocks, found {len(_ALL_BLOCKS)}; "
        f"the Markdown extractor may be misconfigured."
    )
