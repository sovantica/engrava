"""Regression guard — Free public docs must not reference removed observability code.

Observability hook protocol, dispatcher, gates, and event dataclasses were
removed from the public engrava package in v0.3.0. Free engrava package docs
may reference snapshot metrics terminology ("metrics snapshot", etc.) but
must not describe the deleted hook protocol or its associated types.
"""

from __future__ import annotations

from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"

# Symbols deleted in v0.3.0 — must never appear in Free docs.
FORBIDDEN_SYMBOLS = (
    "EngravaObservabilityHooksProtocol",
    "ObservabilityDispatcher",
    "ObservabilityGates",
    "DefaultObservabilityHooks",
    "register_observability_hook",
    "domain.protocols.observability_hooks",
    "domain.models.observability",
)


def test_docs_do_not_reference_deleted_observability_symbols() -> None:
    """Walk all .md files under engrava/docs/ and assert no forbidden symbol."""
    for md_path in DOCS_ROOT.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        for symbol in FORBIDDEN_SYMBOLS:
            assert symbol not in text, (
                f"{md_path.relative_to(DOCS_ROOT.parent)} references "
                f"removed symbol {symbol!r}; observability hooks are no longer "
                f"part of the public engrava package as of v0.3.0. "
                f"Update docs to remove the reference."
            )


def test_benchmarks_directory_not_present() -> None:
    """Free engrava repo must not contain a benchmarks/ directory post v0.3.0."""
    benchmarks_dir = DOCS_ROOT.parent / "benchmarks"
    assert not benchmarks_dir.exists(), (
        f"engrava/benchmarks/ directory should not exist (benchmark suite was "
        f"extracted from this package in v0.3.0). Found: {benchmarks_dir}. "
        f"Run: rm -rf engrava/benchmarks/"
    )
