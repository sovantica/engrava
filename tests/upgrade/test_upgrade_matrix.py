"""Smoke tests for end-to-end user database upgrades."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.upgrade.fixtures import run_upgrade_path, venv_python_path


def test_venv_python_path_points_to_expected_location(tmp_path: Path) -> None:
    """Helper should resolve the virtualenv Python path on the active platform."""
    python_path = venv_python_path(tmp_path / "venv")

    if os.name == "nt":
        assert python_path == tmp_path / "venv" / "Scripts" / "python.exe"
    else:
        assert python_path == tmp_path / "venv" / "bin" / "python"


@pytest.mark.skipif(
    os.environ.get("ENGRAVA_RUN_UPGRADE_MATRIX") != "1",
    reason="Upgrade matrix smoke test is enabled only in dedicated runs.",
)
def test_upgrade_preserves_data_and_core_commands(tmp_path: Path) -> None:
    """Upgrade an existing DB in-place and verify read/write behavior survives."""
    repository_root = Path(__file__).resolve().parents[2]
    from_spec = os.environ["ENGRAVA_UPGRADE_FROM_SPEC"]
    to_spec = os.environ.get("ENGRAVA_UPGRADE_TO_SPEC", ".")
    from_editable = os.environ.get("ENGRAVA_UPGRADE_FROM_EDITABLE", "0") == "1"
    to_editable = os.environ.get("ENGRAVA_UPGRADE_TO_EDITABLE", "1") == "1"

    db_path = tmp_path / "upgrade.db"
    snapshot_path = tmp_path / "upgrade.snapshot.jsonl"

    run_upgrade_path(
        from_spec=from_spec,
        to_spec=to_spec,
        repository_root=repository_root,
        from_editable=from_editable,
        to_editable=to_editable,
        db_path=db_path,
        snapshot_path=snapshot_path,
    )

    assert db_path.exists()
    assert snapshot_path.exists()
