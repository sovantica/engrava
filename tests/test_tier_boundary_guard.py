"""Cognitive-boundary guard tests — dreaming path must not import LLM libraries.

Static assertions that verify the dreaming consolidation path
(including ``_build_clusters``, ``_create_reflections``,
``_lpa_clusters``, and the keyphrase extraction helpers in the
``dreaming_keyphrases`` sibling module) is free of any LLM API
dependency.

Rationale: core infrastructure must not import or call LLM APIs;
REFLECTION content is *structural* (centroid + keyword union + n-gram
TF-IDF + regex entities) — no LLM prose.  Any LLM summary
capability would live in a separate opt-in extension, never in
this package.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

_EXTENSIONS_DIR = Path(__file__).parent.parent / "src" / "engrava" / "extensions"
_DREAMING_PATH = _EXTENSIONS_DIR / "dreaming.py"
_DREAMING_KEYPHRASES_PATH = _EXTENSIONS_DIR / "dreaming_keyphrases.py"
_DREAMING_REFLECTION_CONTENT_PATH = _EXTENSIONS_DIR / "dreaming_reflection_content.py"

# This is a memory database — it does not import LLM-framework runtimes.
# The dreaming consolidation path stays free of any LLM SDK or framework.
_FORBIDDEN_IMPORT_PREFIXES = frozenset(
    {
        "openai",
        "anthropic",
        "google.generativeai",
        "google.ai",
        "ollama",
        "llm",
        "huggingface_hub",
        "transformers",
        "litellm",
        "langchain",
    }
)


def _iter_imports(tree: ast.Module) -> list[tuple[str, str]]:
    """Return ``[(node_kind, module_name), ...]`` for every import in *tree*."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(("import", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(("from", node.module or ""))
    return found


_GUARDED_FILES = (
    ("dreaming.py", _DREAMING_PATH),
    ("dreaming_keyphrases.py", _DREAMING_KEYPHRASES_PATH),
    ("dreaming_reflection_content.py", _DREAMING_REFLECTION_CONTENT_PATH),
)


class TestNoLLMInFreeDreamPath:
    """Cognitive-boundary static guard: dreaming path stays LLM-free."""

    def _source(self, path: Path) -> str:
        assert path.exists(), f"guarded file not found at {path}"
        return path.read_text(encoding="utf-8")

    def _tree(self, path: Path) -> ast.Module:
        return ast.parse(self._source(path))

    # ------------------------------------------------------------------
    # Import guards (apply to every guarded file)
    # ------------------------------------------------------------------

    def test_no_forbidden_imports_anywhere(self) -> None:
        """Every guarded file is free of LLM SDK imports (top-level or inline)."""
        for label, path in _GUARDED_FILES:
            for kind, module in _iter_imports(self._tree(path)):
                for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
                    assert not module.startswith(forbidden), (
                        f"{label} has `{kind} {module}` — "
                        f"violates the cognitive-boundary contract "
                        f"(LLM dependency in core infrastructure)"
                    )

    def test_no_llm_framework_reference(self) -> None:
        """No guarded file names an LLM-framework runtime in its source.

        This is a memory database — the consolidation path does not import
        LLM-framework runtimes, so their package names must not appear in the
        guarded source either (catches dynamic imports / string references
        that the AST import scan above would miss).
        """
        for label, path in _GUARDED_FILES:
            source = self._source(path)
            for framework in ("transformers", "litellm", "langchain"):
                assert framework not in source, (
                    f"{label} references the {framework!r} LLM framework — "
                    f"this is a memory database and does not depend on "
                    f"LLM-framework runtimes"
                )

    def test_importing_dreaming_does_not_pull_llm_sdks(self) -> None:
        """Importing dreaming does not add any LLM SDK to sys.modules."""
        # Record which forbidden modules were already present before import
        before = frozenset(
            m for m in sys.modules if any(m.startswith(p) for p in _FORBIDDEN_IMPORT_PREFIXES)
        )
        import engrava.extensions.dreaming
        import engrava.extensions.dreaming_keyphrases
        import engrava.extensions.dreaming_reflection_content  # noqa: F401

        after = frozenset(
            m for m in sys.modules if any(m.startswith(p) for p in _FORBIDDEN_IMPORT_PREFIXES)
        )
        new_llm_modules = after - before
        assert not new_llm_modules, (
            f"Importing the dreaming path pulled in LLM modules: {new_llm_modules} — "
            f"violates the cognitive-boundary contract"
        )

    # ------------------------------------------------------------------
    # Synchronous / pure helpers
    # ------------------------------------------------------------------

    def test_lpa_clusters_is_synchronous(self) -> None:
        """_lpa_clusters is a synchronous function (no async, no I/O)."""
        from engrava.extensions.dreaming import _lpa_clusters

        assert not inspect.iscoroutinefunction(_lpa_clusters), (
            "_lpa_clusters must be synchronous — async would suggest I/O or LLM dependency"
        )

    def test_lpa_clusters_is_pure_structural(self) -> None:
        """_lpa_clusters operates only on graph adjacency — no embeddings, no LLM."""
        from engrava.extensions.dreaming import _lpa_clusters

        # Should work with purely structural input (no embedding vectors needed)
        adj: dict[str, set[str]] = {
            "a": {"b"},
            "b": {"a"},
            "c": {"d"},
            "d": {"c"},
        }
        clusters = _lpa_clusters(adj)
        assert len(clusters) == 2
        assert all(len(c) == 2 for c in clusters)
