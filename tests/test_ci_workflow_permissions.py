"""Guards the explicit token-scope grant on the PR-triggered CI workflows.

``branch-name-guard.yml`` and ``commitlint.yml`` both run on
``pull_request`` events, where the default ``GITHUB_TOKEN`` grant is broad.
Each workflow declares its own ``permissions:`` block to narrow that grant
instead of relying on the (broader, and repository-setting-dependent)
default. This test does not police the exact scopes granted -- only that
the narrowing block is present and has not been widened to ``write-all``,
which would silently reopen the exposure the block exists to close.
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
