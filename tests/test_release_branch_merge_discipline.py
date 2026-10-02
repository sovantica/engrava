"""Tests for the release-branch merge-discipline check.

The pure decision function (:func:`run_gate`) is exercised directly with
crafted :class:`MergeCommit` values -- no git needed for those cases. The
git-backed cases build a disposable repository under ``tmp_path`` (same
approach as ``test_main_carries_the_released_tag.py``): one arm merges a
feature branch with ``--squash`` and proves GREEN, the other merges the
same feature branch with a real merge commit and proves RED against the
identical content, isolating the merge *shape* as the only variable.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_release_branch_merge_discipline.py"


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_release_branch_merge_discipline.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location(
        "check_release_branch_merge_discipline", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        msg = f"could not load merge-discipline module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _init_disposable_repo(path: Path) -> None:
    """Initialise an empty git repository at ``path`` with a local commit identity."""
    _run(["git", "init", "--quiet", str(path)], cwd=path.parent)
    _run(["git", "config", "user.email", "test@example.invalid"], cwd=path)
    _run(["git", "config", "user.name", "Test"], cwd=path)


def _commit(path: Path, message: str) -> str:
    """Create an empty commit in the repo at ``path`` and return its SHA."""
    _run(["git", "commit", "--quiet", "--allow-empty", "-m", message], cwd=path)
    completed = _run(["git", "rev-parse", "HEAD"], cwd=path)
    return completed.stdout.strip()


class TestRunGatePureLogic:
    """`run_gate` decides PASS/FAIL from an already-collected list -- no git involved."""

    def test_no_merge_commits_passes(self, gate_module: object) -> None:
        passed, messages = gate_module.run_gate([], base="origin/dev", branch="release/v0.7.0")
        assert passed is True
        assert any("PASS" in line for line in messages)

    def test_one_merge_commit_fails_and_names_it(self, gate_module: object) -> None:
        commit = gate_module.MergeCommit(
            sha="a" * 40, subject="Merge branch 'fix/x' into release/v0.7.0"
        )
        passed, messages = gate_module.run_gate(
            [commit], base="origin/dev", branch="release/v0.7.0"
        )
        assert passed is False
        assert any("FAIL" in line and "1 merge commit" in line for line in messages)
        assert any(commit.short_sha in line and commit.subject in line for line in messages)

    def test_multiple_merge_commits_all_named(self, gate_module: object) -> None:
        commits = [
            gate_module.MergeCommit(sha=f"{i}" * 40, subject=f"Merge branch 'x{i}'")
            for i in range(3)
        ]
        passed, messages = gate_module.run_gate(commits, base="origin/dev", branch="release/v0.7.0")
        assert passed is False
        assert any("3 merge commit" in line for line in messages)
        for commit in commits:
            assert any(commit.short_sha in line for line in messages)

    def test_short_sha_is_first_twelve_characters(self, gate_module: object) -> None:
        commit = gate_module.MergeCommit(sha="0123456789abcdef", subject="x")
        assert commit.short_sha == "0123456789ab"


class TestListMergeCommitsAgainstRealHistory:
    """`list_merge_commits` reads real git history -- covered with a disposable repo."""

    def test_linear_squash_history_reports_no_merges(
        self, tmp_path: Path, gate_module: object
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "branch", "dev"], cwd=repo)
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)
        _commit(repo, "feat(core): squashed change one")
        _commit(repo, "fix(core): squashed change two")

        merges = gate_module.list_merge_commits("dev", "release/v0.7.0", repo_root=repo)

        assert merges == []

    def test_a_regular_merge_is_detected(self, tmp_path: Path, gate_module: object) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "branch", "dev"], cwd=repo)
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)
        _run(["git", "checkout", "-b", "fix/pre-insert-seam"], cwd=repo)
        _commit(repo, "Add the pre-insert preparation seam")
        _commit(repo, "Wire the pre-insert seam into create_thought")
        _run(["git", "checkout", "release/v0.7.0"], cwd=repo)
        merge_result = _run(
            ["git", "merge", "--no-ff", "--no-edit", "fix/pre-insert-seam"], cwd=repo
        )
        assert merge_result.returncode == 0, merge_result.stderr

        merges = gate_module.list_merge_commits("dev", "release/v0.7.0", repo_root=repo)

        assert len(merges) == 1
        assert "fix/pre-insert-seam" in merges[0].subject

    def test_identical_content_squash_merged_instead_passes(
        self, tmp_path: Path, gate_module: object
    ) -> None:
        """Same feature-branch content as the RED case above, merged correctly instead."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "branch", "dev"], cwd=repo)
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)
        _run(["git", "checkout", "-b", "fix/pre-insert-seam"], cwd=repo)
        _commit(repo, "Add the pre-insert preparation seam")
        _commit(repo, "Wire the pre-insert seam into create_thought")
        _run(["git", "checkout", "release/v0.7.0"], cwd=repo)
        squash_result = _run(["git", "merge", "--squash", "fix/pre-insert-seam"], cwd=repo)
        assert squash_result.returncode == 0, squash_result.stderr
        _commit(repo, "fix(core): add a pre-insert preparation seam")

        merges = gate_module.list_merge_commits("dev", "release/v0.7.0", repo_root=repo)

        assert merges == []

    def test_the_release_to_dev_merge_itself_is_outside_the_range(
        self, tmp_path: Path, gate_module: object
    ) -> None:
        """release/* -> dev is a regular merge by design and must not be flagged.

        It sits outside ``base..branch`` when ``branch`` is the release
        branch, because ``dev`` is the base -- this is what makes the
        exclusion automatic rather than a special case this gate has to
        know about.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "branch", "dev"], cwd=repo)
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)
        _commit(repo, "feat(core): a clean squashed change")
        _run(["git", "checkout", "dev"], cwd=repo)
        merge_result = _run(["git", "merge", "--no-ff", "--no-edit", "release/v0.7.0"], cwd=repo)
        assert merge_result.returncode == 0, merge_result.stderr

        merges = gate_module.list_merge_commits("dev", "release/v0.7.0", repo_root=repo)

        assert merges == []

    def test_unresolvable_base_raises_gate_input_error(
        self, tmp_path: Path, gate_module: object
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")

        with pytest.raises(gate_module.GateInputError, match="git log --merges"):
            gate_module.list_merge_commits("refs/heads/does-not-exist", "HEAD", repo_root=repo)


class TestCurrentBranch:
    def test_reads_the_checked_out_branch(self, tmp_path: Path, gate_module: object) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)

        assert gate_module.current_branch(repo_root=repo) == "release/v0.7.0"

    def test_detached_head_raises_gate_input_error(
        self, tmp_path: Path, gate_module: object
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        sha = _commit(repo, "chore: base commit")
        _run(["git", "checkout", sha], cwd=repo)

        with pytest.raises(gate_module.GateInputError, match="detached HEAD"):
            gate_module.current_branch(repo_root=repo)


class TestMainEndToEnd:
    def test_exits_zero_on_a_clean_branch(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "branch", "dev"], cwd=repo)
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)
        _commit(repo, "feat(core): squashed change")

        spec = importlib.util.spec_from_file_location(
            "check_release_branch_merge_discipline_e2e_pass", SCRIPT_PATH
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.REPO_ROOT = repo

        assert module.main(["--base", "dev", "--branch", "release/v0.7.0"]) == 0

    def test_exits_one_on_a_branch_with_a_merge(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "chore: base commit")
        _run(["git", "branch", "dev"], cwd=repo)
        _run(["git", "checkout", "-b", "release/v0.7.0"], cwd=repo)
        _run(["git", "checkout", "-b", "fix/x"], cwd=repo)
        _commit(repo, "Some raw fix commit")
        _run(["git", "checkout", "release/v0.7.0"], cwd=repo)
        _run(["git", "merge", "--no-ff", "--no-edit", "fix/x"], cwd=repo)

        spec = importlib.util.spec_from_file_location(
            "check_release_branch_merge_discipline_e2e_fail", SCRIPT_PATH
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.REPO_ROOT = repo

        assert module.main(["--base", "dev", "--branch", "release/v0.7.0"]) == 1
