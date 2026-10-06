"""Layer 4 of the documentation-example tests — documented config defaults.

The compile / phantom-API / behaviour layers catch syntactically-wrong or
nonexistent API, but none of them verify a *documented default value* against
the shipped configuration object. A doc line like ``reflection_boost
(default 1.2)`` compiles fine, names a real field, and runs fine — yet
silently misleads a user who copies the value into their config when the
code actually ships ``1.0``.

This module derives the set of checked fields from the documentation itself,
across the fifty-four fields of ``SearchConfig``, ``DreamingGates``,
``HygienePolicyConfig`` and ``TTLConfig``, via
``tests.docs._documented_defaults_scan``, which finds where ``README.md`` or a
file under ``docs/`` states a default for one of these fields. See that
module's docstring for which statements it compares against the shipped
dataclass and which it records separately.

The expected value is always read from the *code* (``ClassName().field``),
never hard-coded in this file, so the code stays the single source of truth
and the test cannot drift into agreeing with a stale document.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from typing import TYPE_CHECKING

import pytest

import engrava.config as _config_module
from engrava import DreamingGates, HygienePolicyConfig, SearchConfig, TTLConfig
from tests.docs._documented_defaults_scan import (
    Claim,
    ResolvedClaim,
    ScanResult,
    UnparseableClause,
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

# Five fields the derived scan must keep covering — a scan that loses one of
# them has lost coverage however much reach it gains elsewhere.
_ORIGINAL_FIVE: frozenset[tuple[str, str]] = frozenset(
    {
        ("SearchConfig", "reflection_boost"),
        ("HygienePolicyConfig", "gc_restore_window_seconds"),
        ("HygienePolicyConfig", "min_inactivity_age_seconds"),
        ("TTLConfig", "check_every_n_operations"),
        ("DreamingGates", "min_age_cycles"),
    }
)

# A clause is keyed by (doc, the clause text the scanner reports --
# ``UnparseableClause.context``), never by its line number or the fields it
# names, so the key survives lines shifting elsewhere in the doc. The fields
# alone are not unique within a doc: docs/memory-hygiene.md names
# `protected_priorities` in two different clauses, and docs/search.md names
# `collapse_pool_factor` in two.
# See ``_clause_key`` / ``_keyed_unparseable_clauses``.
#
# What this catches and what it does not:
# - This test does not check the values inside an allowlisted clause. Only a
#   dedicated test does, such as
#   ``test_min_and_max_cluster_size_defaults_match_docs``.
# - A change to a clause's words or values, including a value the scanner
#   cannot read, changes its key, so the test fails until someone re-reads
#   the clause and re-keys or removes the entry.
_ClauseKey = tuple[str, str]

# Clauses the scanner names a field in but cannot pair positionally with a
# value (see ``ScanResult.unparseable_clauses``): the count of field mentions
# and value-shaped tokens in the clause disagree, so it is dropped rather
# than compared. Each entry here says why *that* clause's count disagrees, so
# a doc edit that starts producing an unparseable clause somewhere else is
# a live failure -- see ``test_every_unparseable_clause_is_on_a_reasoned_allowlist``.
_UNPARSEABLE_ALLOWLIST: dict[_ClauseKey, str] = {
    (
        "docs/architecture.md",
        (
            "`search_hybrid()` `include_reflections` (default `True`) and `reflection_boost` "
            "(default `None` → uses config)"
        ),
    ): (
        "value tokens: `include_reflections`, `True`, `None` -- `include_reflections` is a "
        "`search_hybrid()` keyword, not a tracked field, and `reflection_boost`'s own value "
        "here is the per-call default `None`, not the field's shipped default -- "
        "3 values against 1 field"
    ),
    (
        "docs/configuration.md",
        (
            "This is separate from > `graph_expansion_enabled` (default `true`), which "
            "controls candidate-pool > widening over `CONSOLIDATED_FROM` edges — the "
            "*ranking* graph signal stays > off until you give `default_graph_weight` (or a "
            "per-call `graph_weight`) a > non-zero value."
        ),
    ): (
        "value tokens: `true`, `CONSOLIDATED_FROM`, `graph_weight` -- `true` is "
        "`graph_expansion_enabled`'s own default, the only real default value in this "
        "clause; the edge type name `CONSOLIDATED_FROM` and the per-call keyword "
        "`graph_weight` are picked up as value-shaped tokens too, and "
        "`default_graph_weight`'s own default (`0.0`) is stated in the preceding clause, "
        "not this one -- 3 values against 2 fields"
    ),
    (
        "docs/configuration.md",
        (
            "Default `signal_weights`: `recency 0.30`, `frequency 0.25`, `confirmation 0.20`, "
            "`confidence 0.15`, `staleness 0.10`"
        ),
    ): (
        "`signal_weights` is a dict; each key's share is written as a compound token "
        "(`recency 0.30`) rather than a bare value, so none register as a value token -- "
        "0 values against 1 field"
    ),
    (
        "docs/data-lifecycle.md",
        (
            "- **A default for the whole store:** `ttl.default_ttl_seconds` in config applies "
            "a default TTL to new thoughts that don't set their own (see [Configuration "
            "→ ttl](configuration.md#ttl))."
        ),
    ): (
        "names `ttl.default_ttl_seconds` while describing its effect in prose, with no "
        "value-shaped token in the clause at all -- 0 values against 1 field"
    ),
    (
        "docs/dreaming.md",
        (
            "- `min_cluster_size` / `max_cluster_size` (defaults `3` / `200`) reject clusters "
            "that are too small or too broad — but not symmetrically: `max_cluster_size` "
            "is applied once, to the **raw** cluster, before eligibility filtering"
        ),
    ): (
        "value tokens: `3`, `200` -- `max_cluster_size` is named twice in the same clause "
        "(once paired with `min_cluster_size`, once again explaining its enforcement "
        "order), so the positional pairing counts 3 field mentions against 2 values -- "
        "checked directly instead, see ``test_min_and_max_cluster_size_defaults_match_docs``"
    ),
    (
        "docs/dreaming.md",
        (
            "Candidate-pool expansion over `CONSOLIDATED_FROM` edges is controlled separately "
            "by `graph_expansion_enabled` (default `true`) and reads those edges only when a "
            "reflection ranks among the top candidates."
        ),
    ): (
        "value tokens: `CONSOLIDATED_FROM`, `true` -- the edge type name "
        "`CONSOLIDATED_FROM` is picked up as a value-shaped token alongside the real "
        "default `true` -- 2 values against 1 field"
    ),
    (
        "docs/evidence-and-conflicts.md",
        (
            "`EdgeType` values are persisted labels: they carry no automatic graph reasoning, "
            "symmetry, transitivity, or conflict propagation. **Ranking is the exception, "
            "twice over.** `graph_expansion_enabled` (default `True`) traverses "
            "`CONSOLIDATED_FROM` edges from top-ranked REFLECTIONs to pull in their source "
            "OBSERVATIONs with a propagated score, regardless of what any other edge type means"
        ),
    ): (
        "value tokens: `EdgeType`, `True`, `CONSOLIDATED_FROM` -- the type name "
        "`EdgeType` and the edge type name `CONSOLIDATED_FROM` are picked up as "
        "value-shaped tokens alongside the real default `True` -- 3 values against 1 field"
    ),
    (
        "docs/glossary.md",
        (
            "Candidate-pool expansion over consolidation edges is a separate step controlled "
            "by `graph_expansion_enabled`, which is on by default and reads those edges only "
            "when a reflection ranks among the top candidates"
        ),
    ): (
        'states the default in plain words ("on by default") with no value-shaped token '
        "in the clause at all -- 0 values against 1 field"
    ),
    (
        "docs/memory-hygiene.md",
        '- its **priority** is listed in `protected_priorities` (default: `("P1",)`).',
    ): (
        'the value is a tuple literal (`("P1",)`), which is not a bare value token shape '
        "-- 0 values against 1 field"
    ),
    (
        "docs/memory-hygiene.md",
        (
            "`protected_priorities` is a **default, not an invariant**: an operator who wants "
            "more aggressive hygiene can set it to `()` so even top-priority thoughts are "
            "eligible"
        ),
    ): (
        "the value is an empty-tuple literal (`()`), which is not a bare value token "
        "shape -- 0 values against 1 field"
    ),
    (
        "docs/search.md",
        (
            "**The table's defaults apply only when a `SearchConfig` is passed to the "
            "store.** `SqliteEngravaCore(conn, ...)` with no `search_config` argument "
            "resolves `default_recency_weight` to `0.0`, not `0.10` — the two are "
            "separate defaults that disagree, and recency is silently inert on a store "
            "built the plain way until you pass a `SearchConfig` explicitly (or an explicit "
            "per-call `recency_weight`)."
        ),
    ): (
        "value tokens: `search_config`, `0.0`, `0.10`, `recency_weight` -- two "
        "contrasting numbers (`0.0`, `0.10`) plus the parameter names `search_config` "
        "and `recency_weight`, all picked up as value-shaped tokens -- "
        "4 values against 1 field"
    ),
    (
        "docs/search.md",
        (
            "When `collapse_key` is set (or the reflection cap is below `1.0`, which the "
            "default `0.3` is), the fallback also widens its own row window by "
            "`search.collapse_pool_factor` beyond `top_k`, the same bounded headroom "
            "collapse and the cap already get from the FTS/vector arms' `fts_top_k` / "
            "`vector_top_k` pools — each is a **minimum** per-arm pool, raised to `top_k` "
            "when smaller, before that widening — so backfill has distinct candidates to "
            "draw from"
        ),
    ): (
        "value tokens: `collapse_key`, `1.0`, `0.3`, `top_k`, `fts_top_k`, "
        "`vector_top_k`, `top_k` -- `search.collapse_pool_factor` is only named here, with "
        "no default value of its own in this clause; `1.0` and `0.3` are the reflection "
        "cap's own comparison threshold and default, `collapse_key`, `fts_top_k`, "
        "`vector_top_k` are unrelated parameter names, and `top_k` is picked up twice "
        "(the fallback's own widen-beyond-`top_k`, and the per-arm floor's "
        "raise-to-`top_k`) -- 7 values against 1 field"
    ),
    (
        "docs/search.md",
        (
            "This does not switch off [reflection-source candidate "
            "expansion](#reflection-source-candidate-expansion), which is controlled "
            "separately by `graph_expansion_enabled` (default `true`) and reads "
            "`CONSOLIDATED_FROM` edges whenever a `REFLECTION` ranks among the top candidates"
        ),
    ): (
        "value tokens: `true`, `CONSOLIDATED_FROM`, `REFLECTION` -- the edge type name "
        "`CONSOLIDATED_FROM` and the type name `REFLECTION` are picked up as "
        "value-shaped tokens alongside the real default `true` -- 3 values against 1 field"
    ),
    (
        "docs/search.md",
        (
            "- To give backfill a deeper pool to draw from, each search arm's candidate "
            "budget is widened by a small, bounded factor **only while** `collapse_key` is "
            "set (configurable as `search.collapse_pool_factor`, default `4`)"
        ),
    ): (
        "value tokens: `collapse_key`, `4` -- the parameter name `collapse_key` is "
        "picked up as a value-shaped token alongside the real default `4` -- "
        "2 values against 1 field"
    ),
    (
        "docs/troubleshooting.md",
        (
            "2. **The confirmation gate.** Unless `allow_zero_confirmation` is `True` (the "
            "default), `confirmation_count` must be at least `min_confirmations` (default `2`)"
        ),
    ): (
        "value tokens: `True`, `confirmation_count`, `2` -- `confirmation_count` is an "
        "attribute the clause discusses but that is not a tracked field, picked up as a "
        "value-shaped token alongside the two real defaults (`True`, `2`) -- "
        "3 values against 2 fields"
    ),
    (
        "docs/upgrade.md",
        (
            "**Your `cluster_quality_cohesion_threshold` was tuned against a different "
            "function, on every provider — check which direction before assuming a "
            "regression, even on `SentenceTransformerProvider`.** The default (`0.40`) was "
            "calibrated on `SentenceTransformerProvider` output"
        ),
    ): (
        "value tokens: `SentenceTransformerProvider`, `0.40`, `SentenceTransformerProvider` "
        "-- the provider class name, mentioned twice, is picked up as a value-shaped "
        "token alongside the real default `0.40` -- 3 values against 1 field"
    ),
}


def _clause_key(clause: UnparseableClause) -> _ClauseKey:
    return (clause.doc, clause.context)


def _keyed_unparseable_clauses(
    clauses: tuple[UnparseableClause, ...],
) -> dict[_ClauseKey, UnparseableClause]:
    """Map each real clause to its ``(doc, context)`` key.

    Two clauses in the same doc with the same normalised text would collide here
    (the second silently overwriting the first in the dict) -- see
    ``test_unparseable_clause_keys_are_unique_per_doc``.
    """
    return {_clause_key(clause): clause for clause in clauses}


def _scan() -> ScanResult:
    return scan_documented_defaults(
        markdown_files(),
        REPO_ROOT,
        field_owner=FIELD_OWNER,
        target_classes=TARGET_CLASSES,
        ambiguous_names=AMBIGUOUS_NAMES,
        section_aliases=SECTION_ALIASES,
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
    """The derived scan must resolve a default for each of the five ``_ORIGINAL_FIVE`` fields."""
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


def test_unparseable_clause_keys_are_unique_per_doc() -> None:
    """``(doc, context)`` must not let two different clauses share one entry.

    Two clauses in one doc with the same text, as the scanner normalises it,
    would share one key, and the dict ``_keyed_unparseable_clauses`` builds
    would keep only one of them. This fails if any two real clauses collide.
    """
    keyed = _keyed_unparseable_clauses(_RESULT.unparseable_clauses)
    assert len(keyed) == len(_RESULT.unparseable_clauses), (
        "two different unparseable clauses collapsed onto the same "
        "(doc, context) key -- widen the key"
    )


def test_every_unparseable_clause_is_on_a_reasoned_allowlist() -> None:
    """Every clause the scanner cannot pair is on the allowlist, with a reason.

    Such a clause is left out of the positional comparison, so that comparison
    cannot catch a wrong default in it. Each one must be named in
    ``_UNPARSEABLE_ALLOWLIST`` with a reason, and no entry may outlive its
    clause. See the comment above the allowlist for how entries are keyed.
    """
    keyed = _keyed_unparseable_clauses(_RESULT.unparseable_clauses)
    unexplained = {key: u for key, u in keyed.items() if key not in _UNPARSEABLE_ALLOWLIST}
    assert not unexplained, [
        f"{u.doc}:{u.line} names {u.fields} against {u.value_count} value token(s) and is "
        f"not on _UNPARSEABLE_ALLOWLIST. Offending text: {u.context!r}"
        for u in unexplained.values()
    ]
    stale = set(_UNPARSEABLE_ALLOWLIST) - set(keyed)
    assert not stale, f"allowlist entries no longer unparseable -- remove or update: {stale}"


def test_min_and_max_cluster_size_defaults_match_docs() -> None:
    """The allowlisted ``min_cluster_size`` / ``max_cluster_size`` clause is checked directly.

    ``min_cluster_size`` / ``max_cluster_size`` in ``docs/dreaming.md`` is on
    ``_UNPARSEABLE_ALLOWLIST`` because the general scanner cannot pair it (see the
    allowlist entry). Being unparseable is not being unchecked: the two values are
    extracted from that exact clause and compared to the shipped defaults by hand, so
    a wrong number here still fails.
    """
    text = (REPO_ROOT / "docs" / "dreaming.md").read_text(encoding="utf-8")
    match = re.search(
        r"`min_cluster_size` / `max_cluster_size` \(defaults `(\d+)` / `(\d+)`\)", text
    )
    assert match is not None, "expected min_cluster_size/max_cluster_size clause not found"
    documented_min, documented_max = (int(g) for g in match.groups())
    gates = DreamingGates()
    assert documented_min == gates.min_cluster_size, (
        f"docs/dreaming.md documents DreamingGates.min_cluster_size as {documented_min}, "
        f"but the shipped default is {gates.min_cluster_size}"
    )
    assert documented_max == gates.max_cluster_size, (
        f"docs/dreaming.md documents DreamingGates.max_cluster_size as {documented_max}, "
        f"but the shipped default is {gates.max_cluster_size}"
    )


def test_scan_reach_is_wider_than_the_old_five_field_registry() -> None:
    """Sanity floor: the derived scan checks far more than the five ``_ORIGINAL_FIVE`` entries.

    Not an exact count — the docs will keep changing wording — just a floor,
    so a change that silently collapses recognition toward zero is caught.
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
        [doc],
        tmp_path,
        field_owner=_FAKE_FIELD_OWNER,
        target_classes=_FAKE_TARGETS,
        ambiguous_names=set(),
        section_aliases={},
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
        [doc],
        tmp_path,
        field_owner=_FAKE_FIELD_OWNER,
        target_classes=_FAKE_TARGETS,
        ambiguous_names=ambiguous,
        section_aliases={},
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
        [doc],
        tmp_path,
        field_owner=_FAKE_FIELD_OWNER,
        target_classes=_FAKE_TARGETS,
        ambiguous_names=ambiguous,
        section_aliases=aliases,
    )
    assert len(result.resolved) == 1
    assert result.resolved[0].matches


_WRONG_TABLE = (
    "| Key | Type | Default | Description |\n"
    "|-----|------|---------|-------------|\n"
    "| `reflection_boost` | `float` | `2.0` | Score multiplier |\n"
)
_WRONG_PROSE = "The `min_age` gate (default `9`) blocks fresh thoughts.\n"


@pytest.mark.parametrize("claim", [_WRONG_TABLE, _WRONG_PROSE], ids=["table", "prose"])
@pytest.mark.parametrize(
    "marker",
    ["    ```", "\t```", "\u00a0```", "    ~~~"],
    ids=["four-spaces", "tab", "no-break-space", "four-spaces-tilde"],
)
def test_scanner_reads_a_claim_between_two_lines_that_only_look_like_fences(
    tmp_path: Path,
    marker: str,
    claim: str,
) -> None:
    """Four spaces, a tab or a no-break space before the backticks is not a fence.

    The two marker lines are an indented code block (or a paragraph), so the
    claim between them is ordinary text and is read; treating the pair as a
    fence would hide a wrong default from the gate.
    """
    result = _fake_scan(tmp_path, f"{marker}\n\n{claim}\n{marker}\n")

    assert len(result.resolved) == 1
    assert not result.resolved[0].matches


@pytest.mark.parametrize("claim", [_WRONG_TABLE, _WRONG_PROSE], ids=["table", "prose"])
@pytest.mark.parametrize("fence", ["```", "~~~", "   ```", "````"])
def test_scanner_does_not_read_a_claim_inside_a_fence(
    tmp_path: Path,
    fence: str,
    claim: str,
) -> None:
    """Control: text in a real fence is code, whatever it says."""
    result = _fake_scan(tmp_path, f"{fence}\n{claim}{fence}\n")

    assert not result.resolved


@pytest.mark.parametrize("claim", [_WRONG_TABLE, _WRONG_PROSE], ids=["table", "prose"])
def test_scanner_does_not_read_a_claim_after_a_line_that_does_not_close_a_fence(
    tmp_path: Path,
    claim: str,
) -> None:
    """A no-break space after the backticks leaves the fence open, so the claim is inside it."""
    result = _fake_scan(tmp_path, f"```\n```\u00a0\n{claim}```\n")

    assert not result.resolved


def test_scanner_rejects_a_fence_that_is_never_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never closed"):
        _fake_scan(tmp_path, f"```\n{_WRONG_TABLE}")


def test_scanner_does_not_split_a_decimal_number_as_a_clause_boundary(tmp_path: Path) -> None:
    """A decimal point inside a value must not be mistaken for a sentence boundary."""
    result = _fake_scan(tmp_path, "`reflection_boost` defaults to `1.0` in every build.\n")
    assert len(result.resolved) == 1
    assert result.resolved[0].doc_value == 1.0
