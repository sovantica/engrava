"""Layer 4 of the documentation-example tests — documented config defaults.

The compile / phantom-API / behaviour layers catch syntactically-wrong or
nonexistent API, but none of them verify a *documented default value* against
the shipped configuration object. A doc line like ``reflection_boost
(default 1.2)`` compiles fine, names a real field, and runs fine — yet
silently misleads a user who copies the value into their config when the
code actually ships ``1.0``.

This module used to close that gap for five hand-picked fields out of the
fifty-four across ``SearchConfig``, ``DreamingGates``, ``HygienePolicyConfig``
and ``TTLConfig`` — the five involved in the incidents that prompted the
check. It now derives the set of checked fields from the documentation
itself, via ``tests.docs._documented_defaults_scan``: every place ``README.md``
or a file under ``docs/`` states a default for one of these fields is found,
resolved against the shipped dataclass, and checked. See that module's
docstring for how a "documented default" is recognised and why some
statements are counted as unresolved rather than compared.

The expected value is always read from the *code* (``ClassName().field``),
never hard-coded in this file, so the code stays the single source of truth
and the test cannot drift into agreeing with a stale document.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING

import pytest

import engrava.config as _config_module
from engrava import DreamingGates, HygienePolicyConfig, SearchConfig, TTLConfig
from tests.docs._documented_defaults_scan import (
    Claim,
    ResolvedClaim,
    ScanResult,
    derive_ambiguous_names,
    derive_field_owner,
    derive_section_aliases,
    scan_documented_defaults,
)
from tests.docs._md_blocks import REPO_ROOT, markdown_files

if TYPE_CHECKING:
    from pathlib import Path

TARGET_CLASSES: dict[str, type] = {
    "SearchConfig": SearchConfig,
    "DreamingGates": DreamingGates,
    "HygienePolicyConfig": HygienePolicyConfig,
    "TTLConfig": TTLConfig,
}
FIELD_OWNER = derive_field_owner(TARGET_CLASSES)
_ALL_CONFIG_DATACLASSES = [
    obj
    for obj in vars(_config_module).values()
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj)
]
AMBIGUOUS_NAMES = derive_ambiguous_names(TARGET_CLASSES, _ALL_CONFIG_DATACLASSES)
SECTION_ALIASES = derive_section_aliases(_config_module.EngravaConfig, TARGET_CLASSES)

# The five fields the old hand-kept registry covered. The derived scan must
# keep covering them — a derived list that loses coverage the hand-kept list
# had is a regression however much reach it gains elsewhere.
_ORIGINAL_FIVE: frozenset[tuple[str, str]] = frozenset(
    {
        ("SearchConfig", "reflection_boost"),
        ("HygienePolicyConfig", "gc_restore_window_seconds"),
        ("HygienePolicyConfig", "min_inactivity_age_seconds"),
        ("TTLConfig", "check_every_n_operations"),
        ("DreamingGates", "min_age_cycles"),
    }
)


def _scan() -> ScanResult:
    return scan_documented_defaults(
        markdown_files(),
        REPO_ROOT,
        FIELD_OWNER,
        TARGET_CLASSES,
        AMBIGUOUS_NAMES,
        SECTION_ALIASES,
    )


# Scanning is pure and the docs tree does not change during a test run, so
# scanning once and sharing the result across every test below keeps a
# full-repo Markdown walk from happening once per parametrised case.
_RESULT = _scan()


def _claim_id(claim: Claim) -> str:
    return f"{claim.doc}:{claim.line}:{claim.cls.__name__}.{claim.field}"


@pytest.mark.parametrize(
    "resolved",
    _RESULT.resolved,
    ids=[_claim_id(r.claim) for r in _RESULT.resolved],
)
def test_documented_default_matches_shipped(resolved: ResolvedClaim) -> None:
    """Each documented default the scanner could resolve equals the shipped value."""
    assert resolved.matches, (
        f"{resolved.claim.doc}:{resolved.claim.line} documents "
        f"{resolved.claim.cls.__name__}.{resolved.claim.field} as {resolved.doc_value!r}, "
        f"but the shipped default is {resolved.shipped!r}. "
        f"Offending text: {resolved.claim.context!r}"
    )


def test_no_documented_default_names_a_nonexistent_field() -> None:
    """A documented default for a field a class does not have is a live defect, not a skip.

    Documentation describing a removed or renamed option is a real defect —
    this is that class, arriving through this scanner's own recognition of
    ``ClassName.field`` mentions.
    """
    assert not _RESULT.nonexistent, [
        f"{n.doc}:{n.line} documents `{n.cls_name}.{n.field_name}`, which does not exist "
        f"on {n.cls_name}. Offending text: {n.context!r}"
        for n in _RESULT.nonexistent
    ]


def test_original_five_fields_remain_covered() -> None:
    """The derived scan must not lose the five fields the old hand-kept list covered."""
    reached = {(r.claim.cls.__name__, r.claim.field) for r in _RESULT.resolved}
    missing = _ORIGINAL_FIVE - reached
    assert not missing, f"the derived scan no longer resolves a default for: {sorted(missing)}"


def test_every_target_field_is_reached_or_explained() -> None:
    """Every field on the four target classes is either checked or a counted, explained skip.

    "Checked" means a resolved claim exists. A field with no resolved claim
    must still show up among the unresolved claims with a reason — silently
    reaching zero claims for a field is exactly the gap this scanner
    replaces the hand-kept list to close.
    """
    reached = {r.claim.field for r in _RESULT.resolved} | {
        u.claim.field for u in _RESULT.unresolved
    }
    missing = set(FIELD_OWNER) - reached
    assert not missing, (
        f"these target fields have no documented-default claim at all, resolved or unresolved: "
        f"{sorted(missing)}"
    )


def test_unresolved_fields_are_only_the_dict_valued_ones() -> None:
    """The only fields this scanner cannot verify are dict-valued (computed, not literal).

    A ``dict`` default (currently just ``HygienePolicyConfig.signal_weights``)
    is documented as several backtick-quoted ``key value`` pairs across one
    sentence, not a single literal — comparing it would mean guessing which
    number belongs to which key. Every other unresolved reason should not
    exist; if one does, that is a real gap in the scanner worth looking at,
    not something to widen this assertion to swallow.
    """
    unresolved_fields = {u.claim.field for u in _RESULT.unresolved}
    dict_valued = {
        name for name, cls in FIELD_OWNER.items() if isinstance(getattr(cls(), name), dict)
    }
    assert unresolved_fields <= dict_valued, (
        f"unresolved for a non-dict-valued field (unexpected — should have matched or been "
        f"a known dict exception): {sorted(unresolved_fields - dict_valued)}"
    )


def test_scan_reach_is_wider_than_the_old_five_field_registry() -> None:
    """Sanity floor: the derived scan checks far more than the old five entries.

    Not an exact count — the docs will keep changing wording — just a floor
    well below the ~70 claims resolved at the time this test was written, so
    a change that silently collapses recognition back toward zero is caught.
    """
    assert len(_RESULT.resolved) >= 40, (
        f"only {len(_RESULT.resolved)} documented defaults resolved; expected the derived "
        f"scan to reach well beyond the old hand-kept registry of 5"
    )


def test_registry_derivation_is_nonempty() -> None:
    """Guard against the derivation inputs themselves silently emptying (vacuous pass)."""
    assert len(FIELD_OWNER) == 54
    assert len(TARGET_CLASSES) == 4


# ---------------------------------------------------------------------------
# Unit tests for the scanner itself, against small synthetic doc trees.
#
# These do not touch the real documentation; they exist to pin the
# recognition and resolution rules described in
# ``_documented_defaults_scan``'s module docstring against regressions, and
# to demonstrate failability directly (see
# ``test_scanner_fails_a_documented_default_that_disagrees_with_the_code``)
# without mutating a real doc file as part of the suite.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _FakeSearchConfig:
    reflection_boost: float = 1.0
    max_neighbors: int = 5


@dataclasses.dataclass(frozen=True)
class _FakeGates:
    enabled: bool = True
    min_age: int = 1


_FAKE_TARGETS: dict[str, type] = {
    "_FakeSearchConfig": _FakeSearchConfig,
    "_FakeGates": _FakeGates,
}
_FAKE_FIELD_OWNER = derive_field_owner(_FAKE_TARGETS)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _fake_scan(tmp_path: Path, doc_text: str) -> ScanResult:
    doc = _write(tmp_path, "doc.md", doc_text)
    return scan_documented_defaults(
        [doc], tmp_path, _FAKE_FIELD_OWNER, _FAKE_TARGETS, ambiguous_names=set(), section_aliases={}
    )


def test_scanner_matches_a_correct_table_default(tmp_path: Path) -> None:
    """Control: a table stating the true default must not fire."""
    result = _fake_scan(
        tmp_path,
        "| Key | Type | Default | Description |\n"
        "|-----|------|---------|-------------|\n"
        "| `reflection_boost` | `float` | `1.0` | Score multiplier |\n",
    )
    assert len(result.resolved) == 1
    assert result.resolved[0].matches


def test_scanner_fails_a_documented_default_that_disagrees_with_the_code(tmp_path: Path) -> None:
    """A documented default that disagrees with the shipped value resolves to a non-match.

    This is the scanner-level proof of failability: a real
    ``test_documented_default_matches_shipped`` parametrised on this claim
    would fail exactly here.
    """
    result = _fake_scan(
        tmp_path,
        "| Key | Type | Default | Description |\n"
        "|-----|------|---------|-------------|\n"
        "| `reflection_boost` | `float` | `2.0` | Score multiplier |\n",
    )
    assert len(result.resolved) == 1
    assert not result.resolved[0].matches
    assert result.resolved[0].shipped == 1.0
    assert result.resolved[0].doc_value == 2.0


def test_scanner_matches_a_correct_parenthetical_default(tmp_path: Path) -> None:
    """Control: the parenthetical prose form stating the true default must not fire."""
    result = _fake_scan(tmp_path, "The `min_age` gate (default `1`) blocks fresh thoughts.\n")
    assert len(result.resolved) == 1
    assert result.resolved[0].matches


def test_scanner_matches_a_correct_inline_kv_default(tmp_path: Path) -> None:
    """Control: the inline ``field = value`` prose form must not fire when correct."""
    result = _fake_scan(tmp_path, "Opt in with `min_age = 1` (the default).\n")
    assert len(result.resolved) == 1
    assert result.resolved[0].matches


def test_scanner_flags_a_default_for_a_field_that_does_not_exist(tmp_path: Path) -> None:
    """A class-qualified mention of a nonexistent field is a live defect, never a silent skip."""
    result = _fake_scan(
        tmp_path, "`_FakeSearchConfig.removed_field` (default `1.0`) no longer exists.\n"
    )
    assert len(result.nonexistent) == 1
    assert result.nonexistent[0].field_name == "removed_field"
    assert not result.resolved
    assert not result.unresolved


def test_scanner_does_not_guess_across_a_call_site_none_default(tmp_path: Path) -> None:
    """A per-call parameter default of ``None`` must not be read as the field's own default."""
    result = _fake_scan(
        tmp_path,
        "`reflection_boost` (default `None` -> uses config) is a call-time override.\n",
    )
    assert not result.resolved
    assert not result.unresolved
    assert len(result.null_skips) == 1


def test_scanner_pairs_multiple_fields_and_values_positionally(tmp_path: Path) -> None:
    """Two fields and two values on one clause pair in the order they appear."""
    result = _fake_scan(
        tmp_path,
        "New `_FakeSearchConfig` fields (`reflection_boost`, `max_neighbors`) "
        "default to (`1.0`, `5`).\n",
    )
    matched = {r.claim.field: r.doc_value for r in result.resolved}
    assert matched == {"reflection_boost": 1.0, "max_neighbors": 5}
    assert all(r.matches for r in result.resolved)


def test_scanner_skips_an_ambiguous_bare_name_outside_its_section(tmp_path: Path) -> None:
    """A field shared with an out-of-scope class is not matched without scope or qualification."""
    ambiguous = {"enabled"}
    doc = _write(
        tmp_path,
        "doc.md",
        "### `unrelated`\n\n| Key | Default |\n|---|---|\n| `enabled` | `false` |\n",
    )
    result = scan_documented_defaults(
        [doc], tmp_path, _FAKE_FIELD_OWNER, _FAKE_TARGETS, ambiguous, section_aliases={}
    )
    assert not result.resolved
    assert len(result.ambiguous_skips) == 1


def test_scanner_accepts_an_ambiguous_bare_name_inside_its_resolved_section(tmp_path: Path) -> None:
    """The same ambiguous name is accepted once a heading resolves the section to its class."""
    ambiguous = {"enabled"}
    aliases: dict[str, type] = {"gates": _FakeGates}
    doc = _write(
        tmp_path,
        "doc.md",
        "### `gates`\n\n| Key | Default |\n|---|---|\n| `enabled` | `true` |\n",
    )
    result = scan_documented_defaults(
        [doc], tmp_path, _FAKE_FIELD_OWNER, _FAKE_TARGETS, ambiguous, aliases
    )
    assert len(result.resolved) == 1
    assert result.resolved[0].matches


def test_scanner_does_not_split_a_decimal_number_as_a_clause_boundary(tmp_path: Path) -> None:
    """A decimal point inside a value must not be mistaken for a sentence boundary."""
    result = _fake_scan(tmp_path, "`reflection_boost` defaults to `1.0` in every build.\n")
    assert len(result.resolved) == 1
    assert result.resolved[0].doc_value == 1.0
