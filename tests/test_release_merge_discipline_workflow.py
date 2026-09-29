"""Ties ``release-merge-discipline.yml`` to the script it exists to run.

``scripts/check_release_branch_merge_discipline.py`` has its own tests, but
those call the script directly. They cannot tell whether the workflow that
is supposed to invoke it does so in a way that works on a real clone. This
test reads the workflow, then executes the very command it contains against
a disposable origin/clone pair: a release branch carrying a merge commit
must fail, and the same content as one squash commit must pass.

It proves the command line works against a clone layout. It does not prove
what GitHub Actions does with the workflow; a hosted run is what shows that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._shell_command import script_argv
from tests._workflow_yaml import load_workflow_text

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-merge-discipline.yml"
SCRIPT_RELATIVE = "scripts/check_release_branch_merge_discipline.py"
SCRIPT_PATH = REPO_ROOT / SCRIPT_RELATIVE
RELEASE_BRANCH = "release/v9.9.9"


def _load_workflow() -> dict[Any, Any]:
    return load_workflow_text(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    assert "on" in workflow, "the trigger key must be the string 'on'"
    return workflow["on"]


def _steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    return workflow["jobs"]["merge-discipline"]["steps"]


def _one_line(command: object) -> object:
    # ``script_argv`` reads one line, so a line that ends in a backslash is joined to the next.
    return command.replace("\\\n", " ") if isinstance(command, str) else command


def _gate_step(workflow: dict[Any, Any]) -> dict[str, Any]:
    matching = [
        s
        for s in _steps(workflow)
        if script_argv(_one_line(s.get("run")), SCRIPT_RELATIVE) is not None
    ]
    assert len(matching) == 1, "exactly one step must run the merge-discipline script"
    return matching[0]


def _git(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _identify(repo: Path) -> None:
    _git(["config", "user.email", "test@example.invalid"], cwd=repo)
    _git(["config", "user.name", "Test"], cwd=repo)


def _write_and_commit(repo: Path, name: str, message: str) -> None:
    (repo / name).write_text(message + "\n", encoding="utf-8")
    _git(["add", name], cwd=repo)
    _git(["commit", "--quiet", "-m", message], cwd=repo)


def _build_clone(tmp_path: Path, *, merge: bool) -> Path:
    """Return a clone of a bare origin whose release branch is squashed or merged.

    Both shapes carry identical file content; only the commit shape differs.
    """
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    clone = tmp_path / "clone"

    _git(["init", "--quiet", "--bare", "--initial-branch=dev", str(origin)], cwd=tmp_path)
    _git(["clone", "--quiet", str(origin), str(seed)], cwd=tmp_path)
    _identify(seed)
    _git(["checkout", "--quiet", "-b", "dev"], cwd=seed)
    _write_and_commit(seed, "base.txt", "chore: base")
    _git(["push", "--quiet", "origin", "dev"], cwd=seed)

    _git(["checkout", "--quiet", "-b", RELEASE_BRANCH, "dev"], cwd=seed)
    _git(["checkout", "--quiet", "-b", "fix/thing", RELEASE_BRANCH], cwd=seed)
    _write_and_commit(seed, "thing-1.txt", "wip one")
    _write_and_commit(seed, "thing-2.txt", "wip two")
    _git(["checkout", "--quiet", RELEASE_BRANCH], cwd=seed)
    if merge:
        _git(
            ["merge", "--quiet", "--no-ff", "-m", "Merge branch 'fix/thing'", "fix/thing"], cwd=seed
        )
    else:
        _git(["merge", "--quiet", "--squash", "fix/thing"], cwd=seed)
        _git(["commit", "--quiet", "-m", "fix: the thing"], cwd=seed)
    _git(["push", "--quiet", "origin", RELEASE_BRANCH], cwd=seed)

    # A fresh clone, like an Actions checkout: remote-tracking refs only, no local
    # release branch, and the script is not tracked (it is copied in like the checkout).
    _git(["clone", "--quiet", str(origin), str(clone)], cwd=tmp_path)
    (clone / "scripts").mkdir()
    shutil.copy(SCRIPT_PATH, clone / SCRIPT_RELATIVE)
    return clone


def _run_workflow_command(clone: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the workflow's own run block in ``clone`` the way the runner would."""
    step = _gate_step(_load_workflow())
    # The step's env values are workflow expressions; supply the event's branch name.
    env = dict.fromkeys(step["env"], RELEASE_BRANCH)
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "python").symlink_to(sys.executable)
    env["PATH"] = f"{shims}{os.pathsep}{os.environ['PATH']}"
    env["HOME"] = str(tmp_path)
    return subprocess.run(  # noqa: S603
        ["bash", "-e", "-c", step["run"]],  # noqa: S607
        cwd=clone,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_workflow_triggers_and_scope() -> None:
    workflow = _load_workflow()
    triggers = _triggers(workflow)
    assert triggers["push"]["branches"] == ["release/**"]
    assert triggers["pull_request"]["branches"] == ["dev"]
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["merge-discipline"]
    # The pull_request trigger cannot filter on the head branch; the job does.
    assert "startsWith(github.head_ref, 'release/')" in job["if"]
    # Pull requests from forks are outside the gate: their head branch does not
    # exist as origin/<branch> in the base repository.
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]
    assert "github.event_name == 'push'" in job["if"]

    checkout = next(
        s for s in _steps(workflow) if s.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == 0


def test_workflow_passes_branch_and_base_explicitly_through_env() -> None:
    step = _gate_step(_load_workflow())
    run = step["run"]
    assert "--branch" in run
    assert "--base" in run
    # No workflow expression may reach the shell line; the branch name goes via env.
    assert "${{" not in run
    assert any("github.head_ref" in str(value) for value in step["env"].values())


@pytest.mark.parametrize(
    ("merge", "expected_exit", "verdict"),
    [(True, 1, "FAIL"), (False, 0, "PASS")],
    ids=["merge-commit", "single-squash-commit"],
)
def test_workflow_command_gates_a_real_clone(
    tmp_path: Path, *, merge: bool, expected_exit: int, verdict: str
) -> None:
    clone = _build_clone(tmp_path, merge=merge)
    completed = _run_workflow_command(clone, tmp_path)
    assert completed.returncode == expected_exit, completed.stdout + completed.stderr
    assert f"MERGE DISCIPLINE GATE: {verdict}" in completed.stdout
