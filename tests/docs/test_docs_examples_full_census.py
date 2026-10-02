"""Layer 8 of the documentation-example tests — every fenced block, every language.

Every other layer in this suite asks the Markdown extractor for one specific
language: ``python`` (layers 1-5), ``bash`` (layer 6), ``yaml`` (layer 7).

This module groups every fenced block across ``README.md`` and ``docs/`` by its
*exact* info string (see ``extract_all_fenced_blocks`` for why exact, not
prefix, matching is required here: prefix matching would make the empty string
collect every fenced block instead of only the bare fences) and asserts the
whole set partitions into exactly:

* the ``python`` blocks, already exhaustively partitioned into executed /
  behaviour-asserted / compile-only by ``test_docs_examples_coverage.py``
  (this module does not re-derive that partition, only re-uses its total);
* the ``bash`` blocks, checked against the real CLI or exempt
  (``test_docs_shell_examples.py``);
* the ``yaml`` blocks, checked against the real config classes or exempt
  (``test_docs_config_examples.py``); and
* every remaining block (a bare fence, or one of the small ``text`` / ``sql``
  / ``json`` buckets) — none of these name a checkable engrava claim, so each
  is registered in ``BARE_AND_MISC_BLOCKS`` with a reason from the closed
  ``ExemptionReason`` vocabulary: a usage-grammar placeholder (a syntax
  template, not a concrete example), an architecture diagram / directory tree
  / error transcript, a scoring formula or algorithm sketch, a concrete bare
  MindQL query, or raw SQL run directly against the database file. Each reason
  is the block's *true* category, not the nearest available one — a formula
  is not filed as a diagram, and a grammar template is not filed as "not an
  invocation".

A block whose location none of these partitions covers, or a bare block that
is not registered, fails ``test_every_fenced_block_is_classified_exactly_once``
below — the same no-silent-gap guarantee the ``python`` census gives, extended
to the whole file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.docs import _md_blocks
from tests.docs import test_docs_examples_coverage as _python_layer
from tests.docs import test_docs_shell_examples as _bash_layer
from tests.docs._md_blocks import (
    REPO_ROOT,
    CodeBlock,
    ExemptionReason,
    all_blocks_by_exact_language,
    extract_all_fenced_blocks,
    extract_exact_fenced_blocks,
    extract_python_blocks,
    markdown_files,
)

if TYPE_CHECKING:
    from pathlib import Path
from tests.docs.test_docs_config_examples import (
    _ALL_YAML_BLOCKS,
    EXEMPT_YAML_BLOCKS,
)
from tests.docs.test_docs_config_examples import (
    _exempt_locations as _yaml_exempt_locations,
)

# The only fence info strings the docs are known to use today. A block whose
# exact info string is not in this set is not routed to any partition below --
# see test_no_fence_uses_an_unrecognised_info_string. Extend this set
# deliberately when a genuinely new fenced language is added to the docs.
_KNOWN_FENCE_LANGUAGES = frozenset({"python", "bash", "yaml", "text", "sql", "json", ""})

# Every fenced block that is neither `python`, `bash`, nor `yaml`: a bare fence
# (no info string) or one of the small `text` / `sql` / `json` buckets. None of
# these name a checkable engrava claim (a real command, flag, or config key),
# so each is registered here with a reason instead of a checker.
#
# Entries are (markdown_path, exact_language, anchor_substring, reason). The
# anchor must appear in exactly one block of that language within its file.
BARE_AND_MISC_BLOCKS: tuple[tuple[str, str, str, ExemptionReason], ...] = (
    (
        "README.md",
        "",
        "COUNT thoughts WHERE lifecycle_status = 'ACTIVE'",
        ExemptionReason.MINDQL_QUERY,
    ),
    (
        "docs/architecture.md",
        "",
        "Extensions / Embeddings / MindQL",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    ("docs/architecture.md", "", "final_score = w", ExemptionReason.FORMULA_OR_PSEUDOCODE),
    (
        "docs/architecture.md",
        "",
        "Dream: REFLECTION thoughts from clusters",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    ("docs/cli.md", "", "mutually exclusive", ExemptionReason.DIAGRAM_OR_TRANSCRIPT),
    (
        "docs/concepts.md",
        "",
        "CONSOLIDATED_FROM  (created by dreaming)",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    ("docs/concurrency.md", "", "sequence contention", ExemptionReason.DIAGRAM_OR_TRANSCRIPT),
    ("docs/dreaming.md", "", "confirmation_count grows", ExemptionReason.DIAGRAM_OR_TRANSCRIPT),
    (
        "docs/dreaming.md",
        "",
        "edge_weight_factor",
        ExemptionReason.FORMULA_OR_PSEUDOCODE,
    ),
    (
        "docs/dreaming.md",
        "",
        "final_score[C] +=",
        ExemptionReason.FORMULA_OR_PSEUDOCODE,
    ),
    ("docs/extensions.md", "", "002_add_tags.sql", ExemptionReason.DIAGRAM_OR_TRANSCRIPT),
    (
        "docs/guides/agent-memory.md",
        "",
        "record the action taken",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/known-limitations.md",
        "",
        "sqlite3.OperationalError: not authorized",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/memory-hygiene.md",
        "",
        "purging the vector index",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/memory-hygiene.md",
        "",
        "eviction_score = keep_score",
        ExemptionReason.FORMULA_OR_PSEUDOCODE,
    ),
    (
        "docs/mindql.md",
        "",
        "[EXPLAIN] SELECT <raw read-only SQL>",
        ExemptionReason.USAGE_GRAMMAR_PLACEHOLDER,
    ),
    (
        "docs/mindql.md",
        "",
        "FIND edges WHERE edge_type = 'ASSOCIATED' LIMIT 5",
        ExemptionReason.MINDQL_QUERY,
    ),
    (
        "docs/mindql.md",
        "",
        "priority ASC, created_cycle DESC LIMIT 10",
        ExemptionReason.MINDQL_QUERY,
    ),
    ("docs/mindql.md", "", "LIMIT 20 OFFSET 40", ExemptionReason.MINDQL_QUERY),
    ("docs/mindql.md", "", "created_cycle IN (1, 2, 3)", ExemptionReason.MINDQL_QUERY),
    ("docs/mindql.md", "", "AND source = 'x'", ExemptionReason.MINDQL_QUERY),
    ("docs/mindql.md", "", "COUNT edges", ExemptionReason.MINDQL_QUERY),
    (
        "docs/mindql.md",
        "",
        "SELECT thought_id, priority, essence FROM thought WHERE thought_type = 'BELIEF' LIMIT 20",
        ExemptionReason.MINDQL_QUERY,
    ),
    (
        "docs/mindql.md",
        "",
        "EXPLAIN SELECT thought_id FROM thought WHERE lifecycle_status = 'ACTIVE'",
        ExemptionReason.MINDQL_QUERY,
    ),
    (
        "docs/mindql.md",
        "",
        "valid_between '2026-01-01T00:00:00+00:00' '2026-12-31T00:00:00+00:00'",
        ExemptionReason.MINDQL_QUERY,
    ),
    (
        "docs/search.md",
        "",
        "ordered by edge.weight DESC (deterministic)",
        ExemptionReason.FORMULA_OR_PSEUDOCODE,
    ),
    (
        "docs/troubleshooting.md",
        "",
        "AttributeError: 'tuple' object has no attribute 'keys'",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/troubleshooting.md",
        "",
        "ValueError: 'INSIGHT' is not a valid ThoughtType",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/troubleshooting.md",
        "",
        "referential integrity violation",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    # `text` blocks -- diagrams and transcripts, same as the bare ones above.
    (
        "docs/concepts.md",
        "text",
        "ARCHIVED --restore_thought()--> ACTIVE",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/data-lifecycle.md",
        "text",
        "ARCHIVED --restore_thought()--> ACTIVE",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/cli.md",
        "text",
        "strand them in it",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/data-lifecycle.md",
        "text",
        "strand them in it",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/evidence-and-conflicts.md",
        "text",
        "existing claim --CONTESTED_BY--> challenging claim",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    ("docs/tutorial.md", "text", "Stored 4 notes.", ExemptionReason.DIAGRAM_OR_TRANSCRIPT),
    (
        "docs/upgrade.md",
        "text",
        "a private attribute such as '_dimension' does not",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/audit-trail.md",
        "text",
        "Restore refused: snapshot line 2 collides",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/backup-and-recovery.md",
        "text",
        "Restore refused: snapshot line 2 collides",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/upgrade.md",
        "text",
        "Restore refused: snapshot line 2 collides",
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    # `sql` blocks -- raw SQL run directly against the SQLite file, not through
    # any engrava-owned surface.
    (
        "docs/data-lifecycle.md",
        "sql",
        "VACUUM INTO 'copy.db'",
        ExemptionReason.RAW_SQL_ILLUSTRATION,
    ),
    (
        "docs/guides/migrating-from-other-memory.md",
        "sql",
        "json_extract(metadata_json",
        ExemptionReason.RAW_SQL_ILLUSTRATION,
    ),
    # `json` block -- a sample hygiene decision transcript.
    (
        "docs/memory-hygiene.md",
        "json",
        '"mechanism": "hygiene"',
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    # `json` blocks -- sample `--json` output transcripts for the one-shot
    # memory verbs (remember / recall / link) and their shared error object.
    (
        "docs/cli.md",
        "json",
        '"error": "invalid_edge_type"',
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/cli.md",
        "json",
        '"schema": "engrava.cli.remember.v1"',
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/cli.md",
        "json",
        '"schema": "engrava.cli.recall.v1"',
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
    (
        "docs/cli.md",
        "json",
        '"schema": "engrava.cli.link.v1"',
        ExemptionReason.DIAGRAM_OR_TRANSCRIPT,
    ),
)


def _unique_block(rel: str, language: str, anchor: str) -> CodeBlock:
    path = REPO_ROOT / rel
    matches = [b for b in extract_exact_fenced_blocks(path, language) if anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"anchor {anchor!r} matched {len(matches)} {language!r}-language blocks in "
            f"{rel} (want exactly 1); update BARE_AND_MISC_BLOCKS in {__file__}.",
        )
    return matches[0]


def _bare_and_misc_locations() -> dict[str, ExemptionReason]:
    return {
        _unique_block(rel, language, anchor).location: reason
        for rel, language, anchor, reason in BARE_AND_MISC_BLOCKS
    }


def test_no_fence_uses_an_unrecognised_info_string() -> None:
    """Every fenced block's exact info string is one this suite knows how to route.

    Comparing block *locations* alone has a hole: retagging a registered
    ``python`` block to ``python3`` does not change its location, and the
    prefix-based ``python`` scan (``extract_python_blocks``) still matches it
    -- so the block stays "covered" by coincidence, and
    ``test_every_fenced_block_is_classified_exactly_once`` would not notice.
    This asserts directly on the set of exact info strings the docs use, so a
    retagged or genuinely new language is rejected on its own terms.
    """
    unknown = sorted(set(all_blocks_by_exact_language()) - _KNOWN_FENCE_LANGUAGES)
    assert not unknown, (
        f"these fenced blocks use an info string this suite does not route "
        f"anywhere: {unknown}. Add a new partition (or extend an existing "
        f"one) deliberately -- an unrecognised language must never merge "
        f"invisibly into whichever partition happens to match it by prefix."
    )


def test_python_prefix_extraction_matches_exact_extraction() -> None:
    """The prefix-based ``python`` scan and the exact ``python`` scan agree everywhere.

    ``extract_python_blocks`` matches by prefix
    (``stripped.startswith("```python")``), so a ``python3``-tagged block is
    silently absorbed into it too. If the two ever disagree, a block is
    ``python``-prefixed but not exactly ``python`` -- the retagging trap the
    module docstring warns about, closed here independently of location
    bookkeeping.
    """
    for path in markdown_files():
        prefix_locations = {b.location for b in extract_python_blocks(path)}
        exact_locations = {b.location for b in extract_exact_fenced_blocks(path, "python")}
        assert prefix_locations == exact_locations, (
            f"{path}: prefix-based python extraction ({sorted(prefix_locations)}) "
            f"disagrees with exact python extraction ({sorted(exact_locations)}) -- "
            f"a fenced block's info string is python-prefixed but not exactly "
            f"'python' (e.g. 'python3')."
        )


def test_retagging_a_block_to_python3_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `python` block retagged to `python3` is caught by the exactness check.

    `extract_python_blocks` (prefix match) still finds a `python3` block when
    asked for `python`, so naive prefix-based coverage stays green on its own
    -- the exact extractor must disagree (what
    ``test_python_prefix_extraction_matches_exact_extraction`` requires never
    happens in the real docs), and the block's exact info string must be
    outside the known-languages set regardless of where it is located.

    ``REPO_ROOT`` is monkeypatched to ``tmp_path`` for this test only, since
    the extractors require every path to resolve relative to it.
    """
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "retag.md"
    md.write_text('```python3\nprint("hi")\n```\n', encoding="utf-8")

    prefix_matches = extract_python_blocks(md)
    exact_python_matches = extract_exact_fenced_blocks(md, "python")
    exact_all = extract_all_fenced_blocks(md)

    assert len(prefix_matches) == 1  # The trap: prefix matching absorbs it.
    assert len(exact_python_matches) == 0  # Exact matching does not.
    assert exact_all[0][0] == "python3"
    assert "python3" not in _KNOWN_FENCE_LANGUAGES


def test_tilde_fence_is_recognised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``~~~``-fenced block is found by the extractors.

    Markdown lets a fence use three tildes instead of backticks.
    ``extract_all_fenced_blocks`` and ``extract_exact_fenced_blocks`` both
    return a tilde-fenced block, so it is counted like a backtick-fenced one.
    """
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "tilde.md"
    md.write_text("~~~bash\nengrava reindex\n~~~\n", encoding="utf-8")

    exact_all = extract_all_fenced_blocks(md)
    assert len(exact_all) == 1
    assert exact_all[0][0] == "bash"
    assert exact_all[0][1].body == "engrava reindex"
    assert len(extract_exact_fenced_blocks(md, "bash")) == 1


def test_unterminated_fence_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fence opened but never closed raises, instead of vanishing.

    Markdown lets a fence run to the end of the document, but the extractor
    raises ``ValueError`` for one rather than silently dropping the block --
    a census whose denominator can shrink like this is worse than none,
    because it still reports a total.
    """
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "unterminated.md"
    md.write_text("```bash\nengrava reindex\n", encoding="utf-8")

    with pytest.raises(ValueError, match="never closed"):
        extract_all_fenced_blocks(md)


def test_fence_inside_a_blockquote_is_recognised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fence whose every line carries a `>` blockquote prefix is found.

    The blockquote prefix is stripped from the fence's opener, body and
    closer, so the block is found and its body carries no leftover `>`.
    """
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "blockquote.md"
    md.write_text("> ```text\n> Error: something failed.\n> ```\n", encoding="utf-8")

    blocks = extract_all_fenced_blocks(md)
    assert len(blocks) == 1
    info, block = blocks[0]
    assert info == "text"
    assert block.body == "Error: something failed."  # No leftover `>` prefix.


def test_ordinary_fence_keeps_a_literal_leading_angle_bracket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `>` inside an ordinary (non-blockquoted) fence is literal content.

    The fence's own opener carries no `>`, so nothing inside it is stripped:
    this invalid, literal-`>`-prefixed YAML keeps its prefix instead of being
    repaired into valid configuration before any validator sees it.
    """
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "ordinary.md"
    md.write_text("```yaml\n> database:\n>   path: demo.db\n```\n", encoding="utf-8")

    blocks = extract_all_fenced_blocks(md)
    assert len(blocks) == 1
    info, block = blocks[0]
    assert info == "yaml"
    assert block.body == "> database:\n>   path: demo.db"


def test_fence_inside_a_list_inside_a_blockquote_is_recognised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fence nested in a list item inside a blockquote is found.

    The list item's own indentation is just extra whitespace once the
    blockquote marker is stripped; deciding "blockquoted or not" once, from
    the opener, and applying it uniformly handles this without modelling
    list containers specifically.
    """
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "listbq.md"
    md.write_text(
        "> - Some list item.\n>\n>   ```bash\n>   engrava reindex\n>   ```\n",
        encoding="utf-8",
    )

    blocks = extract_all_fenced_blocks(md)
    assert len(blocks) == 1
    info, block = blocks[0]
    assert info == "bash"
    assert block.body == "engrava reindex"


def test_four_character_fence_is_recognised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opener longer than three characters is a valid fence too."""
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "longfence.md"
    md.write_text("````bash\nengrava info\n````\n", encoding="utf-8")

    bash_blocks = extract_exact_fenced_blocks(md, "bash")
    assert len(bash_blocks) == 1
    assert bash_blocks[0].body == "engrava info"


def test_closer_longer_than_opener_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CommonMark only requires the closer be at least as long as the opener."""
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "longcloser.md"
    md.write_text("```bash\nengrava info\n`````\n", encoding="utf-8")

    bash_blocks = extract_exact_fenced_blocks(md, "bash")
    assert len(bash_blocks) == 1
    assert bash_blocks[0].body == "engrava info"


def test_closer_with_trailing_whitespace_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CommonMark allows trailing whitespace after the closer."""
    monkeypatch.setattr(_md_blocks, "REPO_ROOT", tmp_path)
    md = tmp_path / "trailingspace.md"
    md.write_text("```bash\nengrava info\n```   \n", encoding="utf-8")

    bash_blocks = extract_exact_fenced_blocks(md, "bash")
    assert len(bash_blocks) == 1
    assert bash_blocks[0].body == "engrava info"


def test_bare_and_misc_registry_anchors_are_unique() -> None:
    """Every BARE_AND_MISC_BLOCKS anchor binds exactly one block of its language."""
    locations = _bare_and_misc_locations()
    assert len(locations) == len(BARE_AND_MISC_BLOCKS), (
        "two BARE_AND_MISC_BLOCKS entries resolved to the same block location; "
        "each entry must name a distinct block."
    )


def test_every_fenced_block_is_classified_exactly_once() -> None:
    """The full-file no-silent-gap guarantee: every fenced block, any language.

    Combines the four independent partitions -- python (layers 1-5), bash
    (layer 6), yaml (layer 7), and bare/text/sql/json (this module) -- and
    asserts they cover every fenced block in the docs exactly once. A block whose
    location none of them covers, or an unregistered bare fence, fails this
    test.
    """
    by_language = all_blocks_by_exact_language()
    all_locations: set[str] = {b.location for blocks in by_language.values() for b in blocks}

    python_locations = (
        _python_layer._executable_locations()
        | _python_layer._behaviour_locations()
        | _python_layer._compile_only_locations()
    )
    bash_locations = {b.location for b in _bash_layer._ALL_BASH_BLOCKS}
    yaml_locations = {b.location for b in _ALL_YAML_BLOCKS}
    bare_and_misc_locations = set(_bare_and_misc_locations())

    partitions = {
        "python": python_locations,
        "bash": bash_locations,
        "yaml": yaml_locations,
        "bare/text/sql/json": bare_and_misc_locations,
    }

    covered: set[str] = set()
    for name, locations in partitions.items():
        overlap = covered & locations
        assert not overlap, (
            f"blocks classified in both an earlier partition and {name!r}: {overlap}"
        )
        covered |= locations

    uncovered = sorted(all_locations - covered)
    assert not uncovered, (
        "these fenced blocks (of any language, including a bare fence) are not "
        f"classified by any layer of the documentation-example suite: {uncovered}"
    )
    extra = sorted(covered - all_locations)
    assert not extra, (
        f"a partition names blocks the extractor no longer finds (stale registry entries): {extra}"
    )

    by_language_counts = {language: len(blocks) for language, blocks in by_language.items()}
    print(  # noqa: T201 — intentional whole-file census summary for the -s report
        f"\nFull fenced-block census: total={len(all_locations)} by-language={by_language_counts}"
    )
    print(  # noqa: T201
        f"  python={len(python_locations)} bash={len(bash_locations)} "
        f"yaml={len(yaml_locations)} bare/text/sql/json={len(bare_and_misc_locations)}"
    )
    reason_tally: dict[str, int] = {}
    for reason in _bare_and_misc_locations().values():
        reason_tally[reason.value] = reason_tally.get(reason.value, 0) + 1
    for reason_value, count in sorted(reason_tally.items()):
        print(f"  bare/text/sql/json reason[{reason_value}] = {count}")  # noqa: T201


def test_yaml_layer_exemptions_are_the_documented_duplicate_key_examples() -> None:
    """Documents exactly which yaml blocks the checker cannot validate, and why.

    See ``test_docs_config_examples.EXEMPT_YAML_BLOCKS`` -- both entries show
    the same top-level key twice, side by side, to illustrate alternate forms.
    That is a real duplicate key by construction, so checking either whole
    block as one document can only ever report the duplicate, never validate
    either form on its own. This is a statement about current docs, not a
    claim the registry must stay this size forever, so it lives here as its
    own small test rather than as an assertion baked into the shared
    exhaustiveness test above.
    """
    assert len(EXEMPT_YAML_BLOCKS) == 2
    assert {reason for _, _, reason, _ in EXEMPT_YAML_BLOCKS} == {
        ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS,
    }
    assert len(_yaml_exempt_locations()) == 2
