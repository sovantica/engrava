"""Tests for the release-target tag gate (F2: a declared target with no reachable tag).

Like ``tests/test_release_target_gate.py``, this needs no
``TestFailabilityOnRealHistory``-style class guarded by ``skipif``: every
case that touches git at all is built from disposable throwaway
repositories under ``tmp_path``, the same pattern
``tests/test_main_carries_the_released_tag.py`` uses for its
``TestIsAncestorAgainstADisposableRepository`` class -- so every case here
runs unconditionally, including on a shallow, depth-1 CI checkout, without
depending on any tag actually present in this repository's own history.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._shell_command import INTERPRETERS, script_argv
from tests._workflow_yaml import load_workflow_text

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_release_target_was_published.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
SCRIPT_RELATIVE = "scripts/check_release_target_was_published.py"
RELEASE_TRIGGER = {"push": {"branches": ["dev"]}}
WORKFLOW_KEYS = {"name", "on", "permissions", "concurrency", "jobs"}
RELEASE_JOB_KEYS = {"name", "runs-on", "outputs", "steps"}


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_release_target_was_published.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("check_release_target_was_published", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load the release-target tag gate module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_disposable_repo(path: Path) -> None:
    """Initialise an empty git repository at ``path`` with a local commit identity."""
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)  # noqa: S603, S607
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],  # noqa: S607
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)  # noqa: S607
    subprocess.run(
        ["git", "commit", "--quiet", "--allow-empty", "-m", "base commit"],  # noqa: S607
        cwd=path,
        check=True,
    )


def _commit(path: Path, message: str) -> str:
    """Create an empty commit in the repository at ``path`` and return its SHA."""
    subprocess.run(  # noqa: S603
        ["git", "commit", "--quiet", "--allow-empty", "-m", message],  # noqa: S607
        cwd=path,
        check=True,
    )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_target(path: Path, version: object) -> None:
    (path / "release-target.json").write_text(json.dumps({"version": version}))


class TestReadDeclaredTarget:
    def test_no_file_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This gate has no legitimate state where the file is missing but
        # the gate still runs -- see the module docstring. An absent file
        # is a broken checkout or an accidental deletion, not "no target
        # declared", so it fails closed like every other bad input here.
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_a_well_formed_file_returns_its_version(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_target(tmp_path, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        assert gate_module.read_declared_target() == "0.7.0"  # type: ignore[attr-defined]

    def test_malformed_json_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "release-target.json").write_text("{not valid json")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_non_object_json_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "release-target.json").write_text(json.dumps(["0.7.0"]))
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_missing_version_key_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "release-target.json").write_text(json.dumps({"not_version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_non_string_version_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_target(tmp_path, 7)
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_malformed_version_string_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_target(tmp_path, "not-a-version")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_a_trailing_newline_in_the_version_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A trailing newline must be rejected. '$' matches just before a
        # trailing newline, so a '^...$'-anchored pattern under match()
        # would accept "0.7.0\n"; VERSION_RE ends in '\Z' instead.
        _write_target(tmp_path, "0.7.0\n")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_a_memory_error_while_reading_raises_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A declaration file too large to read must produce the clean,
        # fail-closed diagnostic every other unreadable input gets here,
        # not an uncaught MemoryError. MemoryError is simulated directly
        # rather than by actually allocating an oversized file; this unit
        # test only pins that the read stage's failure is converted.
        _write_target(tmp_path, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]

        def _raise_memory_error(self: Path, *args: object, **kwargs: object) -> str:
            msg = "simulated: file too large to read"
            raise MemoryError(msg)

        monkeypatch.setattr(Path, "read_text", _raise_memory_error)
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_a_memory_error_while_decoding_raises_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A MemoryError while decoding must be converted too, not only one
        # while reading (see the read-stage test above): 'json.loads()'
        # builds a tree of Python objects, so it can exhaust memory even
        # after a successful read. MemoryError is simulated directly here
        # rather than reproducing that memory pressure in a unit test.
        _write_target(tmp_path, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]

        def _raise_memory_error(*args: object, **kwargs: object) -> object:
            msg = "simulated: document too large to decode"
            raise MemoryError(msg)

        monkeypatch.setattr(json, "loads", _raise_memory_error)
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_a_symlinked_declaration_file_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A release-target.json that is a symlink to unrelated, well-formed
        # JSON elsewhere on the filesystem must not be followed: the
        # declared version would come from outside the checkout and could
        # match an old tag already in the repository. A symlink is refused
        # outright, whether or not its target is itself well-formed.
        ambient_dir = tmp_path / "ambient-outside-the-checkout"
        ambient_dir.mkdir()
        ambient_file = ambient_dir / "elsewhere.json"
        ambient_file.write_text(json.dumps({"version": "0.6.0"}))

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "release-target.json").symlink_to(ambient_file)

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]

    def test_a_non_regular_declaration_file_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A directory named release-target.json is not a symlink, but it is
        # also not a regular file -- the same "require a regular file"
        # guard must reject it too, not just the symlink shape.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "release-target.json").mkdir()
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]


class TestReadDeclaredTargetBoundary:
    """``read_declared_target()`` has one catch-all boundary, not an enumerated exception list.

    A short, hand-picked list of anticipated exception types
    (``OSError``/``MemoryError`` around ``read_text()``,
    ``json.JSONDecodeError``/``MemoryError`` around ``json.loads()``) would
    not name every way reading and parsing can fail.
    """

    def test_invalid_utf8_raises_a_clean_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A file whose bytes are not valid UTF-8. A UnicodeDecodeError out of
        # read_text() is not OSError and not MemoryError, so a list limited
        # to those would miss it.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "release-target.json").write_bytes(b'{"version": "0.7.0\xff\xfe"}')
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]
        assert "UnicodeDecodeError" in str(excinfo.value)

    def test_a_deeply_nested_json_document_raises_a_clean_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 10,000-level nested JSON document. A RecursionError out of
        # json.loads() is not a json.JSONDecodeError, so such a list would
        # miss it too.
        repo = tmp_path / "repo"
        repo.mkdir()
        nested = "[" * 10_000 + "]" * 10_000
        (repo / "release-target.json").write_text('{"version": ' + nested + "}")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]
        assert "RecursionError" in str(excinfo.value)

    def test_an_unrelated_oversized_integer_field_raises_a_clean_gate_input_error(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A further failure mode outside such a list: a field this script
        # never reads at all -- not "version" -- with a 5,000-digit integer
        # literal. json.loads() itself calls int() on every JSON integer
        # literal in the document, and int() refuses a literal past the
        # interpreter's integer-string conversion limit (4300 digits by
        # default), so this raises ValueError from *inside* json.loads(),
        # before this script's own code runs.
        repo = tmp_path / "repo"
        repo.mkdir()
        huge_literal = "9" * 5000
        (repo / "release-target.json").write_text(
            '{"version": "0.7.0", "unrelated_field": ' + huge_literal + "}"
        )
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]
        assert "ValueError" in str(excinfo.value)

    def test_a_deliberate_gate_input_error_is_not_rewrapped(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The boundary must re-raise a deliberate GateInputError exactly as
        # raised, not fold it into the generic "could not read ..." message
        # -- that would trade a specific diagnostic for a vaguer one.
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "release-target.json").write_text(json.dumps({"not_version": "0.7.0"}))
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError) as excinfo:  # type: ignore[attr-defined]
            gate_module.read_declared_target()  # type: ignore[attr-defined]
        expected_path = repo / "release-target.json"
        assert str(excinfo.value) == f"{expected_path} does not declare a 'version' key"


class TestResolveTagCommitAgainstADisposableRepository:
    """Exercise ``resolve_tag_commit``'s real ``^{commit}`` peel against real git objects."""

    def test_a_tag_on_a_commit_resolves(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.resolve_tag_commit("0.7.0") == head  # type: ignore[attr-defined]

    def test_a_missing_tag_resolves_to_none(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.resolve_tag_commit("0.7.0") is None  # type: ignore[attr-defined]

    def test_a_tag_on_a_blob_resolves_to_none(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tag naming a blob, not a commit, must not count as published.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        blob_sha = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            input="not a commit\n",
        ).stdout.strip()
        subprocess.run(["git", "tag", "v0.7.0", blob_sha], cwd=repo, check=True)  # noqa: S603, S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.resolve_tag_commit("0.7.0") is None  # type: ignore[attr-defined]

    def test_a_git_failure_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tmp_path is a plain directory, not a git repository at all -- git
        # cannot answer "does this ref exist" here, so this must not be
        # folded into the "no such tag" (None) case.
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.resolve_tag_commit("0.7.0")  # type: ignore[attr-defined]

    def test_a_symbolic_ref_resolves_to_none(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A symbolic refs/tags/v0.7.0 pointing at the current branch peels
        # straight to HEAD's own commit, with no real tag object anywhere
        # in the repository, so it must not resolve as a tag.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        current_branch = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(  # noqa: S603
            ["git", "symbolic-ref", "refs/tags/v0.7.0", f"refs/heads/{current_branch}"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.resolve_tag_commit("0.7.0") is None  # type: ignore[attr-defined]


class TestIsAncestorAgainstADisposableRepository:
    """Exercise ``is_ancestor``'s real 0/1/other exit-code path against real git objects."""

    def test_true_and_false_cases_are_told_apart(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(
            ["git", "checkout", "--quiet", "-b", "branch-a"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        commit_a = _commit(repo, "commit on branch-a")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "branch-b", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        commit_b = _commit(repo, "commit on branch-b")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        assert gate_module.is_ancestor(base, commit_b) is True  # type: ignore[attr-defined]
        # commit_a and commit_b are on diverging branches -- neither is an
        # ancestor of the other, and both refs resolve cleanly, so this
        # exercises the real "resolves fine, exit code 1" case.
        assert gate_module.is_ancestor(commit_a, commit_b) is False  # type: ignore[attr-defined]

    def test_a_git_failure_raises(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "REPO_ROOT", tmp_path)  # type: ignore[attr-defined]
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.is_ancestor("deadbeef", "HEAD")  # type: ignore[attr-defined]


class TestClassifyTargetAgainstADisposableRepository:
    """Exercise the composed ``classify_target`` -- a tag alone is not enough for REACHABLE."""

    def test_a_tag_on_head_is_reachable(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        state, commit = gate_module.classify_target("0.7.0")  # type: ignore[attr-defined]
        assert state is gate_module.TagState.REACHABLE  # type: ignore[attr-defined]
        assert commit == head

    def test_asks_symbolic_ref_exactly_once_on_a_passing_classification(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # classify_target() asks git 'git symbolic-ref --quiet
        # refs/tags/v<version>' once, to decide whether to call
        # resolve_tag_commit() at all, and passes that answer to
        # resolve_tag_commit() as a parameter rather than having it ask
        # the identical question again. Instrumenting _run_git and calling
        # classify_target() on a tag reachable from HEAD must record
        # exactly one 'symbolic-ref' invocation.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        calls: list[list[str]] = []
        original_run_git = gate_module._run_git  # type: ignore[attr-defined]

        def _counting_run_git(args: list[str]) -> object:
            calls.append(list(args))
            return original_run_git(args)

        monkeypatch.setattr(gate_module, "_run_git", _counting_run_git)  # type: ignore[attr-defined]

        state, _commit_sha = gate_module.classify_target("0.7.0")  # type: ignore[attr-defined]

        assert state is gate_module.TagState.REACHABLE  # type: ignore[attr-defined]
        symbolic_ref_calls = [call for call in calls if call[0] == "symbolic-ref"]
        assert len(symbolic_ref_calls) == 1, calls

    def test_a_missing_tag_is_not_accepted(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        state, commit = gate_module.classify_target("0.7.0")  # type: ignore[attr-defined]
        assert state is gate_module.TagState.NO_TAG  # type: ignore[attr-defined]
        assert commit is None

    def test_a_tag_on_an_unreachable_commit_is_not_accepted(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tag that exists in the object database on a commit this branch
        # never merged must not count as published.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # 'git init' does not reliably create a branch named "main" (it
        # depends on 'init.defaultBranch'), so name one explicitly rather
        # than assume it, matching the pattern used in
        # tests/test_main_carries_the_released_tag.py.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side-branch", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        side_commit = _commit(repo, "unrelated side commit, never merged")
        subprocess.run(["git", "tag", "v0.7.0", side_commit], cwd=repo, check=True)  # noqa: S603, S607
        subprocess.run(
            ["git", "checkout", "--quiet", "main"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        state, commit = gate_module.classify_target("0.7.0")  # type: ignore[attr-defined]
        assert state is gate_module.TagState.UNREACHABLE  # type: ignore[attr-defined]
        assert commit == side_commit

    def test_a_tag_on_a_blob_is_not_accepted(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        blob_sha = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            input="not a commit\n",
        ).stdout.strip()
        subprocess.run(["git", "tag", "v0.7.0", blob_sha], cwd=repo, check=True)  # noqa: S603, S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        state, commit = gate_module.classify_target("0.7.0")  # type: ignore[attr-defined]
        assert state is gate_module.TagState.NON_COMMIT  # type: ignore[attr-defined]
        assert commit is None


class TestRunGateWithStubbedReads:
    """Drive run_gate() with monkeypatched module functions -- no git involved."""

    def test_a_reachable_target_passes(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: "0.7.0")  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module,
            "classify_target",
            lambda _version: (gate_module.TagState.REACHABLE, "deadbeef"),  # type: ignore[attr-defined]
        )
        passed, messages = gate_module.run_gate()  # type: ignore[attr-defined]
        assert passed is True
        assert any("0.7.0" in line and line.startswith("PASS") for line in messages)

    def test_a_target_with_no_reachable_tag_fails(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: "0.7.0")  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module,
            "classify_target",
            lambda _version: (gate_module.TagState.NO_TAG, None),  # type: ignore[attr-defined]
        )
        passed, messages = gate_module.run_gate()  # type: ignore[attr-defined]
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "0.7.0" in fail_lines[0]


class TestMainAgainstADisposableRepository:
    def test_no_target_file_exits_one_and_reports_the_reason(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A repository with no legitimate case for release-target.json
        # being absent must fail closed, not report a clean no-op.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_a_reachable_target_exits_zero(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 0  # type: ignore[attr-defined]
        assert "RELEASE TARGET TAG GATE: PASS" in capsys.readouterr().out

    def test_a_target_with_no_reachable_tag_exits_one(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "0.7.0" in captured.out
        assert "RELEASE TARGET TAG GATE: FAIL" in captured.out

    def test_a_tag_on_an_unreachable_commit_exits_one(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # 'git init' does not reliably create a branch named "main" (it
        # depends on 'init.defaultBranch'), so name one explicitly rather
        # than assume it, matching the pattern used in
        # tests/test_main_carries_the_released_tag.py.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side-branch", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        side_commit = _commit(repo, "unrelated side commit, never merged")
        subprocess.run(["git", "tag", "v0.7.0", side_commit], cwd=repo, check=True)  # noqa: S603, S607
        subprocess.run(
            ["git", "checkout", "--quiet", "main"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_a_tag_on_a_blob_exits_one(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        blob_sha = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            input="not a commit\n",
        ).stdout.strip()
        subprocess.run(["git", "tag", "v0.7.0", blob_sha], cwd=repo, check=True)  # noqa: S603, S607
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_a_git_failure_is_distinguished_from_a_missing_tag(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Not a git repository at all -- release-target.json is present and
        # well-formed, so the failure has to come from git itself being
        # unable to answer, not from "the tag does not exist". The two must
        # be reported differently: this one goes to stderr as a git-call
        # failure, and the gate never reaches its own PASS/FAIL verdict.
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "failed unexpectedly" in captured.err
        assert "no v0.7.0 tag" not in captured.err
        assert "RELEASE TARGET TAG GATE" not in captured.out

    def test_a_malformed_target_file_exits_one(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        (repo / "release-target.json").write_text("{not valid json")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 1  # type: ignore[attr-defined]

    def test_a_symbolic_tag_ref_exits_one(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A symbolic refs/tags/v0.7.0 pointing at the current branch must
        # not let the gate exit 0: no real tag is present, only a ref that
        # peels to HEAD's own commit.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        current_branch = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(  # noqa: S603
            ["git", "symbolic-ref", "refs/tags/v0.7.0", f"refs/heads/{current_branch}"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_pass_message_states_its_own_boundary(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The PASS message must say what it actually established (git's own
        # view of this repository's history) and name what it does not
        # prove (a PyPI upload, or defence against locally altered git
        # metadata) -- not an unqualified "has been published". Asserting
        # only the exit code here would not pin any of that: deleting every
        # word of the boundary sentence and leaving a bare "PASS" would
        # still satisfy `== 0`.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 0  # type: ignore[attr-defined]
        captured = capsys.readouterr()
        assert "resolves to a commit that git reports as" in captured.out
        assert "reachable from HEAD in this repository" in captured.out
        assert "not a defense against locally altered git metadata" in captured.out
        assert "proof that a tag exists, not" in captured.out
        assert "that this version reached PyPI" in captured.out

    def test_a_missing_tag_message_says_no_tag_exists(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A missing tag, a tag that names no commit, and a tag whose commit is
        # not reachable from HEAD each get their own wording. No tag at all
        # must not be worded like a tag-that-exists-but problem, and must not
        # speculate about PyPI or an announcement, which this script cannot
        # know anything about.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "no tag named v0.7.0 exists" in captured.out
        assert "does not name a commit" not in captured.out
        assert "not reachable from HEAD" not in captured.out
        assert "PyPI" not in captured.out
        assert "announced" not in captured.out

    def test_a_non_commit_tag_message_says_it_does_not_name_a_commit(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        blob_sha = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            input="not a commit\n",
        ).stdout.strip()
        subprocess.run(["git", "tag", "v0.7.0", blob_sha], cwd=repo, check=True)  # noqa: S603, S607
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "does not name a commit" in captured.out
        assert "no tag named v0.7.0 exists" not in captured.out
        assert "not reachable from HEAD" not in captured.out
        assert "PyPI" not in captured.out

    def test_an_unreachable_commit_message_names_the_commit(
        self,
        gate_module: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side-branch", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        side_commit = _commit(repo, "unrelated side commit, never merged")
        subprocess.run(["git", "tag", "v0.7.0", side_commit], cwd=repo, check=True)  # noqa: S603, S607
        subprocess.run(
            ["git", "checkout", "--quiet", "main"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert f"resolves to commit {side_commit}" in captured.out
        assert "not reachable from HEAD" in captured.out
        assert "does not name a commit" not in captured.out
        assert "no tag named v0.7.0 exists" not in captured.out
        assert "PyPI" not in captured.out


class TestWorkflowLoader:
    """The loader reads a plain ``on`` as a string, and refuses a repeated key and a merge key."""

    def test_a_plain_on_key_is_read_as_the_string_on(self) -> None:
        assert list(load_workflow_text("on:\n  push: {}\n")) == ["on"]

    @pytest.mark.parametrize("key", ["true", "True", "yes", "Yes", "On", "ON"])
    def test_true_yes_and_capitalised_on_are_not_read_as_the_trigger_key(self, key: str) -> None:
        loaded = load_workflow_text(f"{key}:\n  push: {{}}\n")

        assert "on" not in loaded

    def test_true_and_false_are_still_booleans(self) -> None:
        loaded = load_workflow_text("a: true\nb: false\nc: True\nd: FALSE\n")

        assert loaded == {"a": True, "b": False, "c": True, "d": False}

    @pytest.mark.parametrize(
        "text",
        [
            "on:\n  pull_request: {}\non:\n  push: {}\n",
            "jobs:\n  a:\n    steps:\n      - run: x\n        run: y\n",
        ],
    )
    def test_a_key_repeated_in_a_mapping_is_refused(self, text: str) -> None:
        with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
            load_workflow_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            "base: &b {x: 1}\nuse:\n  <<: *b\n",
            "jobs:\n  a:\n    steps:\n      - run: x\n        <<: {}\n",
        ],
    )
    def test_a_merge_key_is_refused(self, text: str) -> None:
        with pytest.raises(yaml.constructor.ConstructorError, match="merge key"):
            load_workflow_text(text)


class TestReleaseWorkflowWiring:
    """The tests above call the script; these pin the step that runs it in ``release.yml``.

    The gate exists for the run in which nothing was released, so the step
    must run whether or not the release was published, and its failure must
    fail the workflow. The step is pinned to one plain command with no
    argument and no key but its name. The workflow trigger, the workflow's
    keys and the keys of the job that holds the step are pinned to what they
    are today, so a new trigger or a new key fails here until the pin is
    updated. The step must also sit in the same job as the ``detect`` step.
    """

    @staticmethod
    def _workflow() -> dict[str, Any]:
        return load_workflow_text(WORKFLOW_PATH.read_text(encoding="utf-8"))

    @classmethod
    def _jobs(cls) -> dict[str, dict[str, Any]]:
        jobs = cls._workflow()["jobs"]
        assert isinstance(jobs, dict)
        return jobs

    @classmethod
    def _gate_step(cls) -> tuple[str, dict[str, Any]]:
        matching = [
            (job_name, step)
            for job_name, job in cls._jobs().items()
            for step in job.get("steps", [])
            if script_argv(step.get("run"), SCRIPT_RELATIVE) is not None
        ]
        assert len(matching) == 1, (
            f"expected exactly one step whose run command executes {SCRIPT_RELATIVE} under "
            f"an interpreter, found {len(matching)}"
        )
        return matching[0]

    def test_the_step_executes_the_script_rather_than_naming_it(self) -> None:
        _, step = self._gate_step()

        argv = script_argv(step["run"], SCRIPT_RELATIVE)

        assert argv is not None
        assert argv[0] in INTERPRETERS
        assert argv[1] == SCRIPT_RELATIVE

    def test_the_script_is_run_with_no_arguments(self) -> None:
        _, step = self._gate_step()

        argv = script_argv(step["run"], SCRIPT_RELATIVE)

        assert argv is not None
        assert argv[2:] == []

    def test_the_step_sets_nothing_but_its_name_and_its_command(self) -> None:
        _, step = self._gate_step()

        assert set(step) <= {"name", "run"}, (
            "a key on this step (if, continue-on-error, shell, ...) can skip it, excuse its "
            f"failure or change what runs; found {sorted(set(step) - {'name', 'run'})}"
        )

    def test_the_workflow_runs_on_a_push_to_the_release_branch_and_on_nothing_else(self) -> None:
        assert self._workflow()["on"] == RELEASE_TRIGGER

    def test_the_job_and_the_workflow_carry_no_key_they_do_not_carry_today(self) -> None:
        job_name, _ = self._gate_step()
        workflow = self._workflow()
        job = self._jobs()[job_name]

        assert set(workflow) <= WORKFLOW_KEYS, (
            "a key on the workflow (env, defaults, ...) can change whether or how the gate step "
            f"runs; check {sorted(set(workflow) - WORKFLOW_KEYS)} against it, then add it here"
        )
        assert set(job) <= RELEASE_JOB_KEYS, (
            "a key on the job (if, needs, continue-on-error, defaults, env, ...) can skip the "
            "gate step, excuse its failure or change what runs; check "
            f"{sorted(set(job) - RELEASE_JOB_KEYS)} against it, then add it here"
        )

    def test_the_step_belongs_to_the_job_that_detects_whether_a_release_was_published(
        self,
    ) -> None:
        job_name, _ = self._gate_step()

        detecting_jobs = [
            name
            for name, job in self._jobs().items()
            if any(step.get("id") == "detect" for step in job.get("steps", []))
        ]

        assert detecting_jobs == [job_name]
