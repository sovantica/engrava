"""Cognitive-boundary guard tests — the extension surface must not import LLM libraries.

Static assertions over **every** module under ``src/engrava/extensions/``: the
extension surface that ships in the wheel stays free of any LLM SDK or
framework dependency.  The guarded file set is *discovered from the directory*,
never listed here, so a module added to ``extensions/`` is covered by
construction rather than by someone remembering to extend a tuple — a guard
that only reads the files it was born with silently stops guarding as the
directory grows, and reads exactly like a working guard in CI.

Rationale: core infrastructure must not import or call LLM APIs; REFLECTION
content is *structural* (centroid + keyword union + n-gram TF-IDF + regex
entities) — no LLM prose.  Any LLM summary capability would live in a separate
opt-in extension, never in this package.
"""

from __future__ import annotations

import ast
import inspect
import keyword
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

_EXTENSIONS_DIR = Path(__file__).parent.parent / "src" / "engrava" / "extensions"

# This is a memory database — it does not import LLM-framework runtimes.
# The extension surface stays free of any LLM SDK or framework.  This is the one
# place the forbidden set is written down: the AST import scan, the source-text
# scan and the import-closure probe below all read it, so adding a runtime here
# tightens all three at once and none of them can drift behind the others.
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

# Imports every module named on the command line in a *fresh* interpreter and
# reports the whole resulting module table as ``name<TAB>origin`` lines.  Two
# properties matter and neither survives running this in-process:
#
# * a fresh process actually executes the import — in a full test session the
#   extension modules are long since imported, so ``import_module`` returns the
#   cache and runs nothing;
# * reporting the *whole* table rather than a delta means a forbidden runtime
#   that some interpreter start-up hook had already loaded cannot be subtracted
#   away, and the reported origins let the caller confirm which files ran.
_IMPORT_PROBE_SOURCE = (
    "import importlib, sys\n"
    "for target in sys.argv[1:]:\n"
    "    importlib.import_module(target)\n"
    "for name, module in sorted(sys.modules.items()):\n"
    "    print(name, getattr(module, '__file__', '') or '', sep='\\t')\n"
)


def _discover_guarded_files(directory: Path) -> tuple[Path, ...]:
    """Return every Python module under *directory*, deepest paths included.

    Args:
        directory: The package directory whose modules are guarded.

    Returns:
        Every ``*.py`` path below *directory*, sorted for a stable test order.

    """
    return tuple(sorted(directory.rglob("*.py")))


def _module_name(path: Path, *, package_root: Path, package: str) -> str:
    """Return the importable dotted name of *path* inside *package*.

    Args:
        path: A ``*.py`` file inside *package_root*.
        package_root: The directory *package* is rooted at.
        package: The dotted name of the package *package_root* holds.

    Returns:
        The dotted module name (``__init__.py`` maps to the package itself).

    """
    parts = path.relative_to(package_root).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((package, *parts))


def _iter_imports(tree: ast.Module) -> list[tuple[str, str]]:
    """Return ``[(node_kind, module_name), ...]`` for every import in *tree*."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(("import", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(("from", node.module or ""))
    return found


def _forbidden_imports(source: str) -> list[str]:
    """Return a rendered description of every LLM-SDK import in *source*.

    Args:
        source: Python source text to scan.

    Returns:
        One ``"import openai"``-style description per violation; empty when the
        source imports no LLM SDK.

    """
    return [
        f"{kind} {module}"
        for kind, module in _iter_imports(ast.parse(source))
        for forbidden in sorted(_FORBIDDEN_IMPORT_PREFIXES)
        if module.startswith(forbidden)
    ]


def _named_runtimes(source: str) -> list[str]:
    """Return every forbidden runtime named anywhere in *source*, as plain text.

    Scans for the same set the import guard uses rather than a chosen subset of
    it, so a runtime added to ``_FORBIDDEN_IMPORT_PREFIXES`` is caught here too.
    Deliberately blunt: it matches a bare substring, which is what lets it see a
    dynamic ``__import__("litellm")`` or an entry-point string that the AST
    import scan cannot.

    Args:
        source: Python source text to scan.

    Returns:
        The forbidden runtime names present in the text, in sorted order.

    """
    return sorted(name for name in _FORBIDDEN_IMPORT_PREFIXES if name in source)


def _modules_loaded_by(
    module_names: list[str],
    *,
    extra_path: tuple[Path, ...] = (),
) -> dict[str, str]:
    """Import *module_names* in a clean interpreter and report its module table.

    Args:
        module_names: Dotted module names to import, in order.
        extra_path: Directories placed ahead of everything on the child's path.

    Returns:
        Every module the child ended up holding, mapped to the file it came from
        (empty string for built-ins and namespace packages).

    """
    search_path = [str(path) for path in extra_path] + [entry for entry in sys.path if entry]
    completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own probe source
        [sys.executable, "-c", _IMPORT_PROBE_SOURCE, *module_names],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(search_path)},
    )
    assert completed.returncode == 0, (  # precondition: the probe ran at all
        f"import probe failed ({completed.returncode}): {completed.stderr.strip()}"
    )
    return dict(line.split("\t", 1) for line in completed.stdout.splitlines())


def _forbidden_among(module_names: Iterable[str]) -> list[str]:
    """Return the names in *module_names* that belong to a forbidden runtime."""
    return sorted(
        name
        for name in module_names
        if any(name.startswith(prefix) for prefix in _FORBIDDEN_IMPORT_PREFIXES)
    )


def _synthesise_importable(root: Path, dotted: str) -> Path:
    """Create an empty importable module named *dotted* under *root*.

    Every intermediate segment becomes a real (non-namespace) package, so a
    dotted entry of ``_FORBIDDEN_IMPORT_PREFIXES`` is as importable as a
    single-segment one and needs no special case at the call site.

    Args:
        root: Directory the module tree is rooted at; passed to the probe as
            *extra_path*, which puts it ahead of the rest of the child's
            ``PYTHONPATH`` and so ahead of site-packages.
        dotted: The dotted module name to create.

    Returns:
        The path of the leaf module file.

    """
    parts = dotted.split(".")
    package_dir = root
    for part in parts[:-1]:
        package_dir = package_dir / part
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
    leaf = package_dir / f"{parts[-1]}.py"
    leaf.write_text("", encoding="utf-8")
    return leaf


def _walk_python_files(directory: Path) -> set[Path]:
    """Return every ``*.py`` reachable under *directory*, without using ``rglob``.

    A second, independent way of saying "the whole directory", so the assertion
    that discovery excludes nothing is not simply the discovery expression
    written out twice.  It follows directory symlinks, which ``rglob`` does not:
    that makes the oracle strictly more inclusive than discovery, so a module
    reachable only through a symlinked directory shows up as a *failure* here
    rather than as a file the guard quietly never reads.
    """
    found: set[Path] = set()
    visited: set[Path] = set()
    for parent, directories, filenames in os.walk(directory, followlinks=True):
        real_parent = Path(parent).resolve()
        if real_parent in visited:
            # A symlink reached a directory already walked.  Skip it entirely:
            # its files were recorded under the path that reached it first, so
            # nothing escapes the guard, and re-recording them under the alias
            # would only manufacture a divergence against a glob that never
            # follows symlinks.  Clearing *directories* stops a cycle spinning.
            directories.clear()
            continue
        visited.add(real_parent)
        found.update(Path(parent) / filename for filename in filenames if filename.endswith(".py"))
    return found


_GUARDED_FILES = _discover_guarded_files(_EXTENSIONS_DIR)
_GUARDED_IDS = [path.relative_to(_EXTENSIONS_DIR).as_posix() for path in _GUARDED_FILES]


class TestGuardedFileDiscovery:
    """The guarded set is derived from the directory, not restated here."""

    def test_extension_modules_are_discovered(self) -> None:
        """Discovery finds the real extension package.

        A broken path would make every guard below iterate an empty set and
        pass while asserting nothing, so the set itself is pinned as non-empty.
        """
        assert _EXTENSIONS_DIR.is_dir(), f"extensions package not found at {_EXTENSIONS_DIR}"
        assert _GUARDED_FILES, f"no modules discovered under {_EXTENSIONS_DIR}"

    def test_discovery_excludes_nothing(self) -> None:
        """Discovery hands back the directory whole, with nothing filtered out.

        This exists to catch the failure mode that replacing a hand-written file
        list invites: quietly dropping an awkward module inside
        ``_discover_guarded_files`` instead of listing it, which restores the
        original defect one layer down and still reads as a derived guard.  The
        comparison walks the tree by a different route than discovery does, so
        it is a second opinion rather than the same expression twice.
        """
        assert set(_GUARDED_FILES) == _walk_python_files(_EXTENSIONS_DIR)

    def test_a_newly_added_module_is_covered_by_construction(self, tmp_path: Path) -> None:
        """A module that did not exist when this file was written is still guarded.

        The counter-case to a hand-listed file set: dropping a new module into
        the directory must extend the guarded set with no edit here.
        """
        (tmp_path / "already_there.py").write_text("x = 1\n", encoding="utf-8")
        before = _discover_guarded_files(tmp_path)

        (tmp_path / "brand_new_extension.py").write_text("y = 2\n", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "deeper_extension.py").write_text("z = 3\n", encoding="utf-8")
        after = _discover_guarded_files(tmp_path)

        assert {p.name for p in after} - {p.name for p in before} == {
            "brand_new_extension.py",
            "deeper_extension.py",
        }

    def test_a_module_behind_a_directory_symlink_is_not_silently_skipped(
        self, tmp_path: Path
    ) -> None:
        """A file reachable only through a symlinked directory is not invisible.

        ``rglob`` does not descend into symlinked directories, so discovery
        alone would never read such a module and the guard would under-cover in
        silence.  The completeness oracle does descend, which turns that into a
        divergence the check above reports.
        """
        package = tmp_path / "package"
        package.mkdir()
        (package / "visible.py").write_text("x = 1\n", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "hidden.py").write_text("y = 2\n", encoding="utf-8")
        (package / "linked").symlink_to(outside, target_is_directory=True)

        discovered = set(_discover_guarded_files(package))
        walked = _walk_python_files(package)

        assert walked - discovered == {package / "linked" / "hidden.py"}

    def test_the_completeness_walk_terminates_on_a_symlink_cycle(self, tmp_path: Path) -> None:
        """Following symlinks must not let a cycle spin forever.

        The oracle descends through directory symlinks so nothing hides behind
        one; a link pointing back at an ancestor would otherwise recurse without
        end and hang the suite instead of reporting anything.  The fixture also
        carries a file reachable *only* through a symlink, so terminating early
        by refusing to follow links at all would fail this too.
        """
        package = tmp_path / "package"
        (package / "nested").mkdir(parents=True)
        (package / "nested" / "leaf.py").write_text("x = 1\n", encoding="utf-8")
        (package / "nested" / "loop").symlink_to(package, target_is_directory=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "only_here.py").write_text("y = 2\n", encoding="utf-8")
        (package / "side").symlink_to(elsewhere, target_is_directory=True)

        walked = _walk_python_files(package)

        assert walked == {
            package / "nested" / "leaf.py",
            package / "side" / "only_here.py",
        }

    def test_module_name_maps_paths_to_importable_names(self, tmp_path: Path) -> None:
        """Path-to-dotted-name mapping handles packages and submodules alike."""
        named = {
            path: _module_name(tmp_path / path, package_root=tmp_path, package="pkg")
            for path in ("__init__.py", "leaf.py", "sub/leaf.py", "sub/__init__.py")
        }
        assert named == {
            "__init__.py": "pkg",
            "leaf.py": "pkg.leaf",
            "sub/leaf.py": "pkg.sub.leaf",
            "sub/__init__.py": "pkg.sub",
        }


class TestGuardDetectors:
    """The detectors the guards run must actually reject a violating source."""

    def test_forbidden_import_statement_is_detected(self) -> None:
        assert _forbidden_imports("import openai\n") == ["import openai"]

    def test_forbidden_from_import_is_detected(self) -> None:
        assert _forbidden_imports("from langchain.chains import LLMChain\n") == [
            "from langchain.chains"
        ]

    def test_forbidden_import_inside_a_function_is_detected(self) -> None:
        """An inline (lazy) import is a dependency too."""
        assert _forbidden_imports("def f():\n    import anthropic\n") == ["import anthropic"]

    def test_clean_source_reports_no_forbidden_import(self) -> None:
        assert _forbidden_imports("import json\nfrom pathlib import Path\n") == []

    def test_runtime_name_in_a_string_is_detected(self) -> None:
        """A dynamic import hides the name in a string; the source scan sees it."""
        assert "litellm" in _named_runtimes('mod = __import__("litellm")\n')

    def test_clean_source_names_no_runtime(self) -> None:
        assert _named_runtimes("import json\n") == []

    def test_the_text_scan_covers_the_whole_forbidden_set(self) -> None:
        """Every forbidden runtime is scanned for as text, not a chosen subset.

        The text scan used to look for three of the ten names, which is the same
        defect as a hand-listed file set: adding a runtime to the import guard
        left the dynamic-import path uncovered for it.
        """
        for runtime in sorted(_FORBIDDEN_IMPORT_PREFIXES):
            assert runtime in _named_runtimes(f'__import__("{runtime}")\n')

    def test_the_text_scan_is_deliberately_lexical(self) -> None:
        """The text scan matches substrings, and that over-match is the point.

        It exists to catch a package name that never appears as an import — in a
        string, an entry point, a config key — so it cannot use package-boundary
        matching without losing the cases it was written for.  ``litellm``
        therefore also reports ``llm``.  The cost is that an unrelated
        identifier containing a forbidden name is rejected too; in this package
        that is a deliberate, cheap strictness, not an oversight.
        """
        assert _named_runtimes("import litellm\n") == ["litellm", "llm"]
        assert _named_runtimes("count = collmate\n") == ["llm"]
        assert _named_runtimes("import sqlite3\nimport numpy\n") == []


class TestNoLLMInFreeExtensions:
    """Cognitive-boundary static guard: the extension surface stays LLM-free."""

    @pytest.mark.parametrize("path", _GUARDED_FILES, ids=_GUARDED_IDS)
    def test_no_forbidden_imports_anywhere(self, path: Path) -> None:
        """Each extension module is free of LLM SDK imports (top-level or inline)."""
        violations = _forbidden_imports(path.read_text(encoding="utf-8"))
        assert not violations, (
            f"{path.name} has {violations} — violates the cognitive-boundary "
            f"contract (LLM dependency in core infrastructure)"
        )

    @pytest.mark.parametrize("path", _GUARDED_FILES, ids=_GUARDED_IDS)
    def test_no_forbidden_runtime_named_in_the_source(self, path: Path) -> None:
        """No extension module names a forbidden runtime in its source at all.

        This is a memory database — the extension surface does not import LLM
        runtimes, so their package names must not appear in the guarded source
        either (catches dynamic imports and string references that the AST
        import scan above cannot see).
        """
        named = _named_runtimes(path.read_text(encoding="utf-8"))
        assert not named, (
            f"{path.name} references the {named} LLM runtime(s) — this is a "
            f"memory database and does not depend on LLM runtimes"
        )

    def test_importing_the_extensions_does_not_pull_llm_sdks(self) -> None:
        """A fresh interpreter that imports every extension holds no LLM SDK.

        The whole module table is checked, not the modules the import added, so
        a runtime some start-up hook had already loaded cannot be subtracted
        away; and each target's reported origin is checked against this
        checkout, so the guard cannot pass by importing some other installed
        copy of the package.
        """
        expected_origin = {
            _module_name(path, package_root=_EXTENSIONS_DIR, package="engrava.extensions"): path
            for path in _GUARDED_FILES
        }
        loaded = _modules_loaded_by(list(expected_origin))

        # Precondition: the probe executed exactly *these* files, resolved, and
        # not a same-named module from some other copy of the package.
        elsewhere = {
            target: loaded.get(target, "<never loaded>")
            for target, path in expected_origin.items()
            if Path(loaded.get(target, "")).resolve() != path.resolve()
        }
        assert not elsewhere, f"import probe resolved targets outside this checkout: {elsewhere}"

        pulled_in = _forbidden_among(loaded)
        assert not pulled_in, (
            f"Importing the extension surface pulled in LLM modules: {pulled_in} — "
            f"violates the cognitive-boundary contract"
        )

    def test_the_import_probe_reports_what_an_import_loads(self, tmp_path: Path) -> None:
        """The counter-case: the probe does see a module an import pulls in.

        Without this, a probe that silently imported nothing would report a
        table with no violations in it and the guard above would pass for the
        wrong reason.  The origin of the imported file is checked too, since
        that is what the guard above relies on to know it read this checkout.
        """
        (tmp_path / "probe_subject.py").write_text("import colorsys\n", encoding="utf-8")

        loaded = _modules_loaded_by(["probe_subject"], extra_path=(tmp_path,))

        assert loaded.get("probe_subject") == str(tmp_path / "probe_subject.py")
        assert "colorsys" in loaded

    def test_the_forbidden_filter_selects_llm_modules_only(self) -> None:
        """The counter-case for the classification the guard above applies."""
        assert _forbidden_among(["json", "engrava.extensions.dreaming", "sqlite3"]) == []
        assert _forbidden_among(["json", "openai.types", "langchain"]) == [
            "langchain",
            "openai.types",
        ]


class TestTheProbeAndTheFilterJoinUp:
    """The two halves above, joined: a real import that a real filter reports.

    ``test_the_import_probe_reports_what_an_import_loads`` shows the probe sees
    what an import pulls in; ``test_the_forbidden_filter_selects_llm_modules_only``
    shows the filter classifies a hand-written list.  Neither drives the path the
    guard actually takes when the boundary is crossed, and depending on a real
    installed runtime to drive it would make the test a statement about the
    environment rather than about the guard.  Which runtimes are present is not
    stable: eight of the ten are absent here, so a violating module would make
    the guard fail on its "the probe ran at all" precondition
    (``ModuleNotFoundError``) rather than on the boundary — while ``transformers``
    and ``huggingface_hub`` *are* importable, arriving through the
    ``embeddings-hf`` extra and transitively through ``sentence-transformers``
    under ``dev``.  Either way the intended path went unexercised.

    These tests therefore synthesise the runtime instead of importing a real one:
    an empty module named after a forbidden prefix, placed through the
    ``extra_path`` seam ahead of the rest of the child's ``PYTHONPATH`` (the
    ``-c`` working-directory entry still precedes it) and so ahead of
    site-packages, imported by a subject module, and then run through the same
    ``_modules_loaded_by`` + ``_forbidden_among`` pair the guard uses.  Both the
    subject's and the forbidden leaf's reported origins are pinned to the
    fixture, so for the two runtimes that *are* installed the assertion cannot be
    satisfied by the real package.
    """

    subject: str = "boundary_probe_subject"
    bystander: str = "boundary_probe_bystander"

    @pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_IMPORT_PREFIXES))
    def test_a_real_import_of_a_forbidden_runtime_is_reported(
        self, forbidden: str, tmp_path: Path
    ) -> None:
        """Importing a forbidden runtime makes the guard's own pair report it.

        Parametrised over the whole forbidden set rather than one hand-picked
        member: picking a literal would be the same defect the derived guards in
        this file exist to remove, and it would leave a runtime added to the set
        later with no evidence that the join works for it.  Dotted entries are
        covered too — they are synthesised as nested packages — so the set needs
        no exclusion list to stay drivable.  A future entry that cannot spell an
        ``import`` statement at all is *reported*, not skipped: it fails on the
        first assertion below naming itself, rather than as an opaque probe crash
        indistinguishable from the very precondition failure this class exists to
        tell apart from a real violation.
        """
        assert all(
            part.isidentifier() and not keyword.iskeyword(part) for part in forbidden.split(".")
        ), (
            f"{forbidden!r} does not spell a valid ``import`` statement, so the "
            f"join below cannot be driven for it"
        )
        assert not _forbidden_among([self.subject]), (
            f"the subject module name {self.subject!r} is itself classified as "
            f"forbidden, which would let this test pass on the subject's own name "
            f"rather than on the separate runtime it imports"
        )
        leaf = _synthesise_importable(tmp_path, forbidden)
        subject_file = tmp_path / f"{self.subject}.py"
        subject_file.write_text(f"import {forbidden}\n", encoding="utf-8")

        loaded = _modules_loaded_by([self.subject], extra_path=(tmp_path,))

        # Preconditions: the probe ran *this* subject and reached *this*
        # synthetic runtime, not a same-named module from elsewhere on the path.
        # The second matters for ``transformers`` and ``huggingface_hub``, which
        # are really installed here: without it those two parametrisations could
        # be satisfied by the real package and would then be asserting a fact
        # about the environment rather than about the fixture.
        assert loaded.get(self.subject) == str(subject_file)
        assert loaded.get(forbidden) == str(leaf)
        assert _forbidden_among(loaded) == [forbidden]

    def test_an_import_outside_the_forbidden_set_is_reported_clean(self, tmp_path: Path) -> None:
        """The non-target case: the same path reports a harmless import as clean.

        The test above pins what the pair says about a module inside the set;
        this one pins what it says about a module outside it, driven through the
        identical fixture and helpers.  Without the pair discriminating in both
        directions a guard that flagged every deployment would look just as
        healthy in CI as one that works.
        """
        assert not _forbidden_among([self.subject, self.bystander]), (
            f"{self.subject!r}/{self.bystander!r} are classified as forbidden, so "
            f"this test would be red for a reason unrelated to what it checks"
        )
        bystander_file = _synthesise_importable(tmp_path, self.bystander)
        subject_file = tmp_path / f"{self.subject}.py"
        subject_file.write_text(f"import {self.bystander}\n", encoding="utf-8")

        loaded = _modules_loaded_by([self.subject], extra_path=(tmp_path,))

        assert loaded.get(self.subject) == str(subject_file)
        assert loaded.get(self.bystander) == str(bystander_file), (
            "the probe did not observe the import at all"
        )
        assert _forbidden_among(loaded) == []


class TestDreamingIsStructural:
    """The consolidation helpers stay pure and synchronous — no I/O, no LLM."""

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
