"""Guards the explicit token-scope grant on the PR-triggered CI workflows.

``branch-name-guard.yml`` and ``commitlint.yml`` both run on
``pull_request`` events, where the default ``GITHUB_TOKEN`` grant is broad.
Each workflow declares its own ``permissions:`` block to narrow that grant
instead of relying on the (broader, and repository-setting-dependent)
default. Neither workflow's steps push, comment, or otherwise write to the
repository or the pull request -- both only read the checkout and lint it --
so the grant each is meant to have is exactly ``contents: read`` and nothing
else. A grant that widens any individual scope to ``write`` reopens the
exposure the block exists to close just as much as the ``write-all``
shorthand does, and is checked here too, not only that shorthand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

_PR_TRIGGERED_WORKFLOWS = (
    "branch-name-guard.yml",
    "commitlint.yml",
)

# The exact permissions block each workflow needs -- not a ceiling on
# read-only scopes, which cost nothing, but a ceiling on every write scope:
# neither workflow pushes, comments, or otherwise writes anything, so
# neither is meant to hold a single "write" entry, and this dict is checked
# for equality, not containment.
_EXPECTED_PERMISSIONS: dict[str, dict[str, str]] = {
    "branch-name-guard.yml": {"contents": "read"},
    "commitlint.yml": {"contents": "read"},
}


def _load_workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("workflow_name", _PR_TRIGGERED_WORKFLOWS)
def test_workflow_declares_a_narrowed_permissions_block(workflow_name: str) -> None:
    workflow = _load_workflow(workflow_name)
    permissions = workflow.get("permissions")
    assert permissions, (
        f"{workflow_name}: top-level 'permissions:' key is missing or empty -- "
        "this workflow would fall back to the default, broader GITHUB_TOKEN grant"
    )
    assert permissions != "write-all", (
        f"{workflow_name}: 'permissions: write-all' grants the broad default "
        "back -- narrow it to only what the workflow's steps need"
    )


@pytest.mark.parametrize("workflow_name", _PR_TRIGGERED_WORKFLOWS)
def test_workflow_permissions_grant_no_write_scope(workflow_name: str) -> None:
    """No individual scope in the block is widened to ``write``, not only ``write-all``."""
    workflow = _load_workflow(workflow_name)
    permissions = workflow.get("permissions")
    assert isinstance(permissions, dict), (
        f"{workflow_name}: expected a mapping of scope -> access level, got {permissions!r}"
    )
    write_scopes = {scope: level for scope, level in permissions.items() if level == "write"}
    assert not write_scopes, (
        f"{workflow_name}: grants write access on {sorted(write_scopes)} -- neither "
        "workflow pushes, comments, or otherwise writes anything, so no scope here "
        "should be 'write'"
    )


@pytest.mark.parametrize("workflow_name", _PR_TRIGGERED_WORKFLOWS)
def test_workflow_permissions_match_exactly_what_it_needs(workflow_name: str) -> None:
    """The declared block is exactly the scopes the workflow needs, no more and no less."""
    workflow = _load_workflow(workflow_name)
    permissions = workflow.get("permissions")
    expected = _EXPECTED_PERMISSIONS[workflow_name]
    assert permissions == expected, (
        f"{workflow_name}: permissions block is {permissions!r}, expected exactly "
        f"{expected!r} -- if the workflow now genuinely needs a different grant, "
        "update _EXPECTED_PERMISSIONS deliberately rather than widening this check"
    )
