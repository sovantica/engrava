"""Tests for the schema-version release gate.

Most cases drive the pure logic functions directly with crafted values —
no git needed. The ``TestFailabilityOnRealHistory`` class is the exception:
it runs the gate against this repository's own tags, because a check like
this is only worth anything once it has actually been seen red as well as
green (see the module docstring in ``check_schema_release_gate.py``).
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_schema_release_gate.py"


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_schema_release_gate.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("check_schema_release_gate", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load schema release gate module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tag_exists(tag: str) -> bool:
    completed = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


REQUIRES_V0_5_0_AND_V0_6_0 = pytest.mark.skipif(
    not (_tag_exists("v0.5.0") and _tag_exists("v0.6.0")),
    reason="requires the v0.5.0 and v0.6.0 tags to be present in this checkout",
)


class TestIsPatchOnlyBump:
    def test_patch_only(self, gate_module: object) -> None:
        assert gate_module.is_patch_only_bump("0.6.0", "0.6.1") is True  # type: ignore[attr-defined]

    def test_minor_bump_is_not_patch_only(self, gate_module: object) -> None:
        assert gate_module.is_patch_only_bump("0.6.0", "0.7.0") is False  # type: ignore[attr-defined]

    def test_major_bump_is_not_patch_only(self, gate_module: object) -> None:
        assert gate_module.is_patch_only_bump("0.6.0", "1.0.0") is False  # type: ignore[attr-defined]

    def test_unchanged_version_is_not_patch_only(self, gate_module: object) -> None:
        assert gate_module.is_patch_only_bump("0.6.0", "0.6.0") is False  # type: ignore[attr-defined]


class TestParseVersion:
    def test_parses_bare_semver(self, gate_module: object) -> None:
        assert gate_module.parse_version("1.2.3") == (1, 2, 3)  # type: ignore[attr-defined]

    def test_rejects_a_leading_v(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("v1.2.3")  # type: ignore[attr-defined]

    def test_rejects_a_prerelease_suffix(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.parse_version("1.2.3-rc.1")  # type: ignore[attr-defined]


class TestSchemaMoved:
    def test_unchanged_sql_and_no_old_constant_is_not_moved(self, gate_module: object) -> None:
        moved = gate_module.schema_moved(  # type: ignore[attr-defined]
            old_sql=20,
            new_sql=20,
            old_constant=None,
            new_constant=20,
        )
        assert moved is False

    def test_changed_sql_is_moved_even_without_an_old_constant(self, gate_module: object) -> None:
        moved = gate_module.schema_moved(  # type: ignore[attr-defined]
            old_sql=18,
            new_sql=20,
            old_constant=None,
            new_constant=20,
        )
        assert moved is True

    def test_changed_constant_is_moved_even_if_sql_somehow_agreed(
        self, gate_module: object
    ) -> None:
        """Either stamp moving is enough -- neither one is trusted alone."""
        moved = gate_module.schema_moved(  # type: ignore[attr-defined]
            old_sql=20,
            new_sql=20,
            old_constant=19,
            new_constant=20,
        )
        assert moved is True


class TestRunGateWithStubbedReads:
    """Drive run_gate() with monkeypatched stamp readers -- no git involved.

    Covers the integrity paths that real history cannot exercise: a
    candidate release missing the constant entirely, and a candidate
    release whose two stamps contradict each other.
    """

    def test_new_ref_missing_the_constant_falls_back_to_sql_and_still_catches_the_violation(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        """Absence of the constant at the candidate ref is not itself a failure.

        It is expected history for any tag predating the constant's
        introduction; the SQL stamp alone must still be enough to catch a
        patch release that moved the schema.
        """
        sql_by_ref = {"v0.6.0": 18, "deadbeef": 20}
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "read_sql_stamp", lambda ref: sql_by_ref[ref]
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "read_core_constant", lambda _ref: None
        )
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            old_tag="v0.6.0",
            new_version="0.6.1",
            new_ref="deadbeef",
        )
        assert passed is False
        assert any(line.startswith("FAIL") for line in messages)

    def test_disagreeing_stamps_at_candidate_release_fails(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        sql_by_ref = {"v0.6.0": 20, "deadbeef": 21}
        constant_by_ref = {"v0.6.0": 20, "deadbeef": 20}
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "read_sql_stamp", lambda ref: sql_by_ref[ref]
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "read_core_constant", lambda ref: constant_by_ref[ref]
        )
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            old_tag="v0.6.0",
            new_version="0.6.1",
            new_ref="deadbeef",
        )
        assert passed is False
        assert any("stamps disagree" in line for line in messages)

    def test_patch_bump_with_no_schema_move_passes(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(gate_module, "read_sql_stamp", lambda _ref: 20)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "read_core_constant", lambda _ref: 20
        )
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            old_tag="v0.6.0",
            new_version="0.6.1",
            new_ref="deadbeef",
        )
        assert passed is True
        assert any(line.startswith("PASS") for line in messages)


@REQUIRES_V0_5_0_AND_V0_6_0
class TestFailabilityOnRealHistory:
    """The acceptance bar this gate exists to meet: seen both green and red.

    v0.5.0 -> v0.6.0 is real, published history in which the core schema
    version moved (18 -> 20) across a minor bump -- that must pass. Neither
    tag defines CORE_SCHEMA_HEAD_VERSION yet (it was introduced after
    v0.6.0 shipped), so this pair also exercises the SQL-stamp fallback for
    real: schema_core.sql's own PRAGMA is the only signal available here,
    and it has to be enough on its own.

    The failing case reuses that exact same schema move but relabels it
    with a synthetic patch-only version pair, since no real release has
    ever actually broken this rule.
    """

    def test_a_real_minor_release_that_moved_the_schema_passes(self, gate_module: object) -> None:
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            old_tag="v0.5.0",
            new_version="0.6.0",
            new_ref="v0.6.0",
        )
        assert passed is True, messages
        assert any(line.startswith("PASS") for line in messages)

    def test_a_synthetic_patch_relabelling_of_the_same_move_fails(
        self, gate_module: object
    ) -> None:
        passed, messages = gate_module.run_gate(  # type: ignore[attr-defined]
            old_tag="v0.5.0",
            new_version="0.5.1",
            new_ref="v0.6.0",
        )
        assert passed is False, messages
        assert any(line.startswith("FAIL") for line in messages)
