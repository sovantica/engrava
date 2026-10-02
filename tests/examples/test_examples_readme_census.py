"""Does ``examples/README.md``'s index name exactly what ``examples/`` ships?

Three shipped files (``agent_loop.py``, ``notes_memory.py``, ``config.yaml``)
were once missing from this page, found by a person reading it rather than by
any gate — the documentation-example suite's scope is ``README.md`` plus
``docs/*.md`` only (see ``tests/docs/_md_blocks.py``), and ``examples/`` sits
outside it entirely. This module is that gate: a small, self-contained census
over ``examples/`` (see ``_examples_census.py`` for why it is its own module
rather than a widening of the shared docs scope), asserting the index and the
directory agree in both directions.
"""

from __future__ import annotations

from tests.examples._examples_census import EXAMPLES_DIR, EXAMPLES_README, compute_census


def test_every_shipped_file_has_an_index_entry() -> None:
    """Every file ``examples/`` ships is linked from ``examples/README.md``."""
    census = compute_census(EXAMPLES_DIR, EXAMPLES_README)
    assert not census.missing_from_index, (
        f"examples/README.md never links: {sorted(census.missing_from_index)}"
    )


def test_every_shipped_file_has_exactly_one_index_entry() -> None:
    """No shipped file is linked from the index more than once."""
    census = compute_census(EXAMPLES_DIR, EXAMPLES_README)
    assert not census.duplicated_in_index, (
        f"linked more than once in examples/README.md: {census.duplicated_in_index}"
    )


def test_index_names_no_file_that_does_not_ship() -> None:
    """The reverse direction: a stale entry naming a deleted file is the same defect."""
    census = compute_census(EXAMPLES_DIR, EXAMPLES_README)
    missing = sorted(census.missing_from_disk)
    assert not missing, f"examples/README.md links a file examples/ does not ship: {missing}"


def test_every_relative_link_resolves() -> None:
    """Every non-http(s) link on the page, not just same-directory ones, resolves."""
    census = compute_census(EXAMPLES_DIR, EXAMPLES_README)
    assert not census.unresolved_relative_links, (
        f"examples/README.md has a relative link that does not resolve: "
        f"{census.unresolved_relative_links}"
    )


def test_every_documented_invocation_names_a_shipped_script() -> None:
    """Every `python examples/<x>.py` line the page prints names a file that ships."""
    census = compute_census(EXAMPLES_DIR, EXAMPLES_README)
    assert not census.dangling_invocations, (
        f"examples/README.md prints `python examples/<x>` for a file that does not ship: "
        f"{census.dangling_invocations}"
    )
