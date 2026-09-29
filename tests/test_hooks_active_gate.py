"""Tests for ``scripts/check_hooks_active.sh`` (``make hooks-active``).

The gate exists to say whether the commit-msg hook is doing its job. Wiring
is not the job: a hook file replaced by an executable ``exit 0`` is wired
and executable, and it checks nothing. So these cases run the script against a
disposable checkout -- a fresh ``git init`` under ``tmp_path`` with a copy of
this repository's ``.githooks``, commitlint configuration and scope vocabulary,
and ``core.hooksPath`` set to that copy. What is at ``.githooks/commit-msg`` and
at the checkout's ``.git/hooks/commit-msg``, the guard the hook chains to,
differs from case to case, and in one case a merge is in progress.

The cases need the same tools the hook needs, so they are skipped when
``bash``, ``git`` or ``node`` is not on ``PATH`` or commitlint is not installed
in the primary checkout (``make install`` installs it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_hooks_active.sh"

_COPIED_FILES = (".commitlintrc.js", "commit-scopes.json")
_ALWAYS_ACCEPTS = "#!/usr/bin/env bash\nexit 0\n"
_ALWAYS_REJECTS = "#!/usr/bin/env bash\necho 'nothing is good enough' >&2\nexit 1\n"


def _clean_git_env() -> dict[str, str]:
    """Return the environment without the inherited ``GIT_*`` variables.

    A test run from inside a git hook inherits ``GIT_DIR`` and other ``GIT_*``
    variables, which would point git at the wrong repository. The user's and the
    system's git configuration are switched off as well.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _primary_node_modules() -> Path | None:
    """Return the primary checkout's ``node_modules`` if it carries commitlint, else None.

    The hook looks for commitlint in the primary checkout, which is the parent
    of the one git-dir every worktree of a repository shares.
    """
    git = shutil.which("git")
    if git is None:
        return None
    completed = subprocess.run(  # noqa: S603
        [git, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_clean_git_env(),
    )
    if completed.returncode != 0:
        return None
    node_modules = Path(completed.stdout.strip()).parent / "node_modules"
    if not os.access(node_modules / ".bin" / "commitlint", os.X_OK):
        return None
    return node_modules


_NODE_MODULES = _primary_node_modules()

pytestmark = pytest.mark.skipif(
    _NODE_MODULES is None or any(shutil.which(tool) is None for tool in ("bash", "git", "node")),
    reason=(
        "needs bash, git and node on PATH and commitlint installed in the primary "
        "checkout's node_modules (run 'make install')"
    ),
)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Build a disposable checkout wired to its own copy of the real hooks."""
    assert _NODE_MODULES is not None
    root = (tmp_path / "checkout").resolve()
    subprocess.run(  # noqa: S603
        ["git", "init", "--quiet", str(root)],  # noqa: S607
        check=True,
        env=_clean_git_env(),
    )
    shutil.copytree(REPO_ROOT / ".githooks", root / ".githooks")
    for name in _COPIED_FILES:
        shutil.copy2(REPO_ROOT / name, root / name)
    (root / "node_modules").symlink_to(_NODE_MODULES, target_is_directory=True)
    subprocess.run(  # noqa: S603
        ["git", "config", "core.hooksPath", str(root / ".githooks")],  # noqa: S607
        cwd=root,
        check=True,
        env=_clean_git_env(),
    )
    return root


def _replace_hook(checkout: Path, body: str) -> Path:
    hook = checkout / ".githooks" / "commit-msg"
    hook.write_text(body, encoding="utf-8")
    hook.chmod(0o755)
    return hook


def _run_gate(checkout: Path, scratch_parent: Path) -> subprocess.CompletedProcess[str]:
    """Run the gate script from the disposable checkout, scratch files under ``scratch_parent``."""
    scratch_parent.mkdir(exist_ok=True)
    env = _clean_git_env()
    env["TMPDIR"] = str(scratch_parent)
    return subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT_PATH)],  # noqa: S607
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestGateExercisesTheHook:
    def test_a_real_hook_passes(self, checkout: Path, tmp_path: Path) -> None:
        result = _run_gate(checkout, tmp_path / "scratch")

        assert result.returncode == 0, result.stderr
        assert "the commit-msg hook accepts a well-formed message" in result.stdout

    def test_a_hook_that_accepts_everything_fails_and_says_so(
        self, checkout: Path, tmp_path: Path
    ) -> None:
        hook = _replace_hook(checkout, _ALWAYS_ACCEPTS)

        result = _run_gate(checkout, tmp_path / "scratch")

        assert result.returncode != 0
        assert "commit-msg hook" in result.stderr
        assert str(hook) in result.stderr
        assert "accepted a message whose type is not allowed" in result.stderr
        assert "wibble(docs)" in result.stderr

    def test_a_hook_that_rejects_everything_fails_and_says_so(
        self, checkout: Path, tmp_path: Path
    ) -> None:
        hook = _replace_hook(checkout, _ALWAYS_REJECTS)

        result = _run_gate(checkout, tmp_path / "scratch")

        assert result.returncode != 0
        assert str(hook) in result.stderr
        assert "rejected a well-formed message" in result.stderr
        assert "docs(docs): describe the thing" in result.stderr
        assert "nothing is good enough" in result.stderr

    def test_a_guard_installed_in_the_hooks_directory_is_part_of_the_hook_under_test(
        self, checkout: Path, tmp_path: Path
    ) -> None:
        # The hook chains to <git-common-dir>/hooks/commit-msg, and a guard
        # there that exits nonzero makes the hook reject the message.
        guard = checkout / ".git" / "hooks" / "commit-msg"
        guard.parent.mkdir(exist_ok=True)
        guard.write_text(_ALWAYS_REJECTS, encoding="utf-8")
        guard.chmod(0o755)

        result = _run_gate(checkout, tmp_path / "scratch")

        assert result.returncode != 0
        assert "rejected a well-formed message" in result.stderr
        assert "nothing is good enough" in result.stderr

    def test_a_merge_in_progress_fails_instead_of_skipping_the_check(
        self, checkout: Path, tmp_path: Path
    ) -> None:
        # The hook skips its own grammar check for a true merge, so the gate
        # cannot exercise that check. The gate must not read that as a passing hook.
        (checkout / ".git" / "MERGE_HEAD").write_text("0" * 40 + "\n", encoding="utf-8")

        result = _run_gate(checkout, tmp_path / "scratch")

        assert result.returncode != 0
        assert "a merge is in progress" in result.stderr

    @pytest.mark.parametrize(
        "hook_body",
        [None, _ALWAYS_ACCEPTS, _ALWAYS_REJECTS],
        ids=["real-hook", "accepts-everything", "rejects-everything"],
    )
    def test_no_scratch_files_are_left_behind(
        self, checkout: Path, tmp_path: Path, hook_body: str | None
    ) -> None:
        if hook_body is not None:
            _replace_hook(checkout, hook_body)
        scratch_parent = tmp_path / "scratch"

        _run_gate(checkout, scratch_parent)

        assert list(scratch_parent.iterdir()) == []
