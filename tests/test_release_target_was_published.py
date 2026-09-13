"""Tests for the release-target-was-published gate (F2: no version was published, silently).

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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_release_target_was_published.py"


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_release_target_was_published.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("check_release_target_was_published", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load release-target-was-published module from {SCRIPT_PATH}"
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
        # Regression for the finding that a tag naming *any* object -- not
        # specifically a commit -- was previously accepted as "published".
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


class TestTargetIsPublishedAgainstADisposableRepository:
    """Exercise the composed ``target_is_published`` -- a tag alone is not enough."""

    def test_a_tag_on_head_is_published(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.target_is_published("0.7.0") is True  # type: ignore[attr-defined]

    def test_a_missing_tag_is_not_published(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.target_is_published("0.7.0") is False  # type: ignore[attr-defined]

    def test_a_tag_on_an_unreachable_commit_is_not_published(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression for the finding that a tag existing anywhere in the
        # object database -- even on a commit this branch never merged --
        # was previously accepted as "published".
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
        assert gate_module.target_is_published("0.7.0") is False  # type: ignore[attr-defined]

    def test_a_tag_on_a_blob_is_not_published(
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
        assert gate_module.target_is_published("0.7.0") is False  # type: ignore[attr-defined]


class TestRunGateWithStubbedReads:
    """Drive run_gate() with monkeypatched module functions -- no git involved."""

    def test_a_published_target_passes(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: "0.7.0")  # type: ignore[attr-defined]
        monkeypatch.setattr(gate_module, "target_is_published", lambda _version: True)  # type: ignore[attr-defined]
        passed, messages = gate_module.run_gate()  # type: ignore[attr-defined]
        assert passed is True
        assert any("0.7.0" in line and line.startswith("PASS") for line in messages)

    def test_an_unpublished_target_fails(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_declared_target", lambda: "0.7.0")  # type: ignore[attr-defined]
        monkeypatch.setattr(gate_module, "target_is_published", lambda _version: False)  # type: ignore[attr-defined]
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
        # Regression for the finding that deleting release-target.json made
        # the gate pass: a repository with no legitimate case for the file
        # being absent must fail closed, not report a clean no-op.
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        exit_code = gate_module.main([])  # type: ignore[attr-defined]
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_a_published_target_exits_zero(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        subprocess.run(["git", "tag", "v0.7.0"], cwd=repo, check=True)  # noqa: S607
        _write_target(repo, "0.7.0")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 0  # type: ignore[attr-defined]

    def test_an_unpublished_target_exits_one(
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
        assert "FAIL" in captured.out

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
        assert "RELEASE TARGET WAS-PUBLISHED GATE" not in captured.out

    def test_a_malformed_target_file_exits_one(
        self, gate_module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        (repo / "release-target.json").write_text("{not valid json")
        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        assert gate_module.main([]) == 1  # type: ignore[attr-defined]
