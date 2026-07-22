"""Layer 4 of the documentation-example tests — documented config defaults.

The compile / phantom-API / behaviour layers catch syntactically-wrong or
nonexistent API, but none of them verify a *documented default value* against
the shipped configuration object. A doc line like
``reflection_boost (default 1.2)`` compiles fine, names a real field, and runs
fine — yet silently misleads a user who copies the value into their config when
the code actually ships ``1.0``.

This module closes that gap. For every config field whose default is quoted in
the docs, it asserts the documented value equals the runtime default on the
shipped dataclass — and that the value actually appears at the documented
location, so the test fails loudly if either the docs or this registry drift.

To add a new checked default: append a row to ``DOCUMENTED_DEFAULTS`` with the
config class, field name, and the doc files that state it. The expected value is
read from the *code*, never hard-coded here, so the code stays the single source
of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from engrava import DreamingGates, HygienePolicyConfig, SearchConfig, TTLConfig
from tests.docs._md_blocks import REPO_ROOT


@dataclass(frozen=True)
class DocumentedDefault:
    """A config default that the documentation quotes.

    Attributes:
        config_factory: Zero-arg callable returning a default config instance.
        field: The attribute whose default is documented.
        doc_files: Doc paths (relative to repo root) that state the default.
        label: Human-readable id for test parametrisation.

    """

    config_factory: type
    field: str
    doc_files: tuple[str, ...]
    label: str


# Each entry pairs a documented config default with the shipped dataclass that
# owns it. The expected value is read from the dataclass at runtime (never
# hard-coded), so the assertion is "the docs match the code", with the code as
# the single source of truth.
DOCUMENTED_DEFAULTS: tuple[DocumentedDefault, ...] = (
    DocumentedDefault(
        config_factory=SearchConfig,
        field="reflection_boost",
        doc_files=("docs/search.md", "docs/architecture.md"),
        label="SearchConfig.reflection_boost",
    ),
    DocumentedDefault(
        config_factory=TTLConfig,
        field="check_every_n_operations",
        doc_files=("docs/data-lifecycle.md",),
        label="TTLConfig.check_every_n_operations",
    ),
    DocumentedDefault(
        config_factory=DreamingGates,
        field="min_age_cycles",
        doc_files=("docs/troubleshooting.md",),
        label="DreamingGates.min_age_cycles",
    ),
    DocumentedDefault(
        config_factory=HygienePolicyConfig,
        field="min_inactivity_age_seconds",
        doc_files=("docs/configuration.md",),
        label="HygienePolicyConfig.min_inactivity_age_seconds",
    ),
    DocumentedDefault(
        config_factory=HygienePolicyConfig,
        field="gc_restore_window_seconds",
        doc_files=("docs/configuration.md",),
        label="HygienePolicyConfig.gc_restore_window_seconds",
    ),
)


def _format_default(value: object) -> str:
    """Render a default value the way the docs quote it (``1.0`` not ``1``)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return str(value)


@pytest.mark.parametrize(
    "spec",
    DOCUMENTED_DEFAULTS,
    ids=[d.label for d in DOCUMENTED_DEFAULTS],
)
def test_documented_default_matches_shipped(spec: DocumentedDefault) -> None:
    """Each documented config default equals the shipped runtime default."""
    shipped = getattr(spec.config_factory(), spec.field)
    rendered = _format_default(shipped)

    # The value the docs MUST state (e.g. ``1.0``), in the backtick form the
    # docs use for inline code.
    expected_token = f"`{rendered}`"

    # A line documents the CONFIG-FIELD default (not a method-param default)
    # when it names the field, says "default", and quotes a numeric literal.
    # Lines whose default is ``None`` / "uses config" describe the per-call
    # method parameter, not the config field — they are skipped.
    numeric_default = re.compile(r"`?\d+(?:\.\d+)?`?")

    for rel in spec.doc_files:
        path = REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")

        field_default_lines = [
            ln
            for ln in text.splitlines()
            if spec.field in ln
            and "default" in ln.lower()
            and "none" not in ln.lower()
            and numeric_default.search(ln) is not None
        ]
        assert field_default_lines, (
            f"{rel} names no numeric documented default for {spec.field!r}; "
            f"update DOCUMENTED_DEFAULTS or the doc."
        )
        for line in field_default_lines:
            assert expected_token in line or f"= {rendered}" in line or f": {rendered}" in line, (
                f"{rel} documents {spec.field} with a default that does not match "
                f"the shipped value {rendered!r} (from {spec.config_factory.__name__}). "
                f"Offending line: {line.strip()!r}. Fix the doc to state {rendered!r}."
            )


def test_registry_is_nonempty() -> None:
    """Guard against the registry silently emptying (vacuous pass)."""
    assert len(DOCUMENTED_DEFAULTS) >= 3, (
        "DOCUMENTED_DEFAULTS shrank unexpectedly; documented config defaults "
        "would no longer be verified against the shipped code."
    )
