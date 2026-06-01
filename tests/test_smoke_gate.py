"""Tests for the pre-publish smoke gate.

The synthetic benchmark itself is slow (~5 minutes), so the unit tests
patch :func:`check_smoke_gate.run_synthetic_benchmark` to return a
synthetic payload rather than driving the real CLI. A separate test
exercises the floor evaluator directly with crafted payloads to pin
each individual floor's failure path.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_smoke_gate.py"


@pytest.fixture
def smoke_gate_module() -> object:
    """Load ``scripts/check_smoke_gate.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("check_smoke_gate", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load smoke gate module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAYLOAD_ALL_PASS = {
    "binding": [
        {
            "label": "AC-9a synthesis coverage",
            "value": 0.85,
            "threshold": 0.80,
            "passed": True,
            "rule_text": ">= 0.80",
        },
        {
            "label": "AC-9b direct neutrality",
            "value": 0.03,
            "threshold": 0.05,
            "passed": True,
            "rule_text": "<= 0.05",
        },
        {
            "label": "AC-8  sanity tolerance",
            "value": 0.02,
            "threshold": 0.05,
            "passed": True,
            "rule_text": "<= 0.05",
        },
        {
            "label": "AC-8b sanity (boost=1.0)",
            "value": 0.02,
            "threshold": 0.05,
            "passed": True,
            "rule_text": "<= 0.05",
        },
    ],
    "passed": True,
}


def _payload_with_override(label: str, value: float) -> dict[str, object]:
    """Return a passing payload with one floor's value replaced."""
    binding = [dict(entry) for entry in PAYLOAD_ALL_PASS["binding"]]  # type: ignore[arg-type]
    for entry in binding:
        if entry["label"] == label:
            entry["value"] = value
            entry["passed"] = False
    return {"binding": binding, "passed": False}


class TestEvaluateSyntheticFloors:
    """Direct tests for the floor evaluator."""

    def test_all_pass(self, smoke_gate_module: object) -> None:
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(  # type: ignore[attr-defined]
            PAYLOAD_ALL_PASS,
        )
        assert passed is True
        assert failures == []

    def test_synthesis_coverage_below_floor(self, smoke_gate_module: object) -> None:
        payload = _payload_with_override("AC-9a synthesis coverage", 0.70)
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(payload)  # type: ignore[attr-defined]
        assert passed is False
        assert any("AC-9a synthesis coverage" in line and "< floor" in line for line in failures)

    def test_direct_neutrality_above_ceiling(self, smoke_gate_module: object) -> None:
        payload = _payload_with_override("AC-9b direct neutrality", 0.12)
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(payload)  # type: ignore[attr-defined]
        assert passed is False
        assert any("AC-9b direct neutrality" in line and "> ceiling" in line for line in failures)

    def test_sanity_neutrality_above_ceiling(self, smoke_gate_module: object) -> None:
        payload = _payload_with_override("AC-8  sanity tolerance", 0.20)
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(payload)  # type: ignore[attr-defined]
        assert passed is False
        assert any("sanity tolerance" in line and "> ceiling" in line for line in failures)

    def test_sanity_boost_above_ceiling(self, smoke_gate_module: object) -> None:
        payload = _payload_with_override("AC-8b sanity (boost=1.0)", 0.30)
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(payload)  # type: ignore[attr-defined]
        assert passed is False
        assert any("AC-8b sanity (boost=1.0)" in line and "> ceiling" in line for line in failures)

    def test_missing_binding_section(self, smoke_gate_module: object) -> None:
        passed, failures = smoke_gate_module.evaluate_synthetic_floors({})  # type: ignore[attr-defined]
        assert passed is False
        assert any("missing the 'binding' section" in line for line in failures)

    def test_missing_label(self, smoke_gate_module: object) -> None:
        payload = {
            "binding": [
                entry
                for entry in PAYLOAD_ALL_PASS["binding"]  # type: ignore[union-attr]
                if entry["label"] != "AC-9a synthesis coverage"
            ],
        }
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(payload)  # type: ignore[attr-defined]
        assert passed is False
        assert any(
            "missing binding result for 'AC-9a synthesis coverage'" in line for line in failures
        )

    def test_non_numeric_value(self, smoke_gate_module: object) -> None:
        payload = _payload_with_override("AC-9a synthesis coverage", float("nan"))
        binding = payload["binding"]
        for entry in binding:  # type: ignore[union-attr]
            if entry["label"] == "AC-9a synthesis coverage":
                entry["value"] = "definitely-not-a-number"  # type: ignore[assignment]
        passed, failures = smoke_gate_module.evaluate_synthetic_floors(payload)  # type: ignore[attr-defined]
        assert passed is False
        assert any("not numeric" in line for line in failures)


class TestCheckRecallAgainstFloor:
    """Direct tests for the LongMemEval recall@5 floor comparator."""

    def test_recall_at_floor_passes(self, smoke_gate_module: object) -> None:
        passed, failures = smoke_gate_module._check_recall_against_floor(  # type: ignore[attr-defined]
            0.30,
            floor=0.30,
        )
        assert passed is True
        assert failures == []

    def test_recall_above_floor_passes(self, smoke_gate_module: object) -> None:
        passed, failures = smoke_gate_module._check_recall_against_floor(  # type: ignore[attr-defined]
            0.38,
            floor=0.30,
        )
        assert passed is True
        assert failures == []

    def test_recall_below_floor_fails(self, smoke_gate_module: object) -> None:
        passed, failures = smoke_gate_module._check_recall_against_floor(  # type: ignore[attr-defined]
            0.25,
            floor=0.30,
        )
        assert passed is False
        assert len(failures) == 1
        assert "0.250" in failures[0]
        assert "0.300" in failures[0]
        assert "<" in failures[0]


class TestEvaluateLongmemevalRecall:
    """Floor-enforcement behaviour of the LongMemEval probe."""

    def test_refuses_to_enforce_uncalibrated_floor(
        self,
        smoke_gate_module: object,
    ) -> None:
        passed, failures = smoke_gate_module.evaluate_longmemeval_recall(floor=None)  # type: ignore[attr-defined]
        assert passed is False
        assert any("not yet calibrated" in line for line in failures)

    def test_committed_floor_is_below_calibrated_aggregate(
        self,
        smoke_gate_module: object,
    ) -> None:
        """The shipped floor must not exceed the observed maintainer aggregate.

        The maintainer calibration on the upstream LongMemEval oracle
        variant in substring mode (top_k=5, dreaming ON) measured
        aggregate_score = 0.3835 on the 498-question subset (two
        questions excluded for the engrava-FTS5 follow-up). The shipped
        floor must sit at or below that value — a higher value would be
        a guess, not an empirical floor.
        """
        floor = smoke_gate_module.LONGMEMEVAL_RECALL_AT_5_FLOOR  # type: ignore[attr-defined]
        assert floor is not None, "LONGMEMEVAL_RECALL_AT_5_FLOOR must be set to a calibrated value"
        assert floor <= 0.3835, f"floor {floor} exceeds the maintainer calibration value 0.3835"

    def test_excluded_question_ids_are_documented(
        self,
        smoke_gate_module: object,
    ) -> None:
        """Every excluded id must be a known LongMemEval-oracle FTS-crasher."""
        excluded = smoke_gate_module.LONGMEMEVAL_EXCLUDED_QUESTION_IDS  # type: ignore[attr-defined]
        # Known crashers identified during calibration. If the set
        # changes, the FTS normaliser follow-up note must change with
        # it.
        assert excluded == frozenset({"352ab8bd", "gpt4_59149c77"})


class TestMain:
    """End-to-end main() behaviour with the synthetic runner patched."""

    def test_main_returns_zero_on_passing_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        def fake_run() -> tuple[int, dict[str, object]]:
            return 0, PAYLOAD_ALL_PASS

        monkeypatch.setattr(smoke_gate_module, "run_synthetic_benchmark", fake_run)  # type: ignore[arg-type]
        rc = smoke_gate_module.main([])  # type: ignore[attr-defined]
        assert rc == 0

    def test_main_returns_one_on_floor_breach(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        breached = _payload_with_override("AC-9a synthesis coverage", 0.50)

        def fake_run() -> tuple[int, dict[str, object]]:
            return 1, breached

        monkeypatch.setattr(smoke_gate_module, "run_synthetic_benchmark", fake_run)  # type: ignore[arg-type]
        rc = smoke_gate_module.main([])  # type: ignore[attr-defined]
        assert rc == 1

    def test_main_returns_one_when_synthetic_runner_reports_failure_even_if_floors_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        """If the runner exits non-zero we must NOT swallow it even if our floor check passes."""

        def fake_run() -> tuple[int, dict[str, object]]:
            return 1, PAYLOAD_ALL_PASS

        monkeypatch.setattr(smoke_gate_module, "run_synthetic_benchmark", fake_run)  # type: ignore[arg-type]
        rc = smoke_gate_module.main([])  # type: ignore[attr-defined]
        assert rc == 1

    def test_main_returns_two_on_missing_extras(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        def fake_run() -> tuple[int, dict[str, object]]:
            return 2, {}

        monkeypatch.setattr(smoke_gate_module, "run_synthetic_benchmark", fake_run)  # type: ignore[arg-type]
        rc = smoke_gate_module.main([])  # type: ignore[attr-defined]
        assert rc == 2

    def test_main_with_include_longmemeval_passes_when_recall_clears_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        """``--include-longmemeval`` exits 0 when the probe reports a passing recall."""

        def fake_synthetic_run() -> tuple[int, dict[str, object]]:
            return 0, PAYLOAD_ALL_PASS

        def fake_lme_eval() -> tuple[bool, list[str]]:
            return True, []

        monkeypatch.setattr(smoke_gate_module, "run_synthetic_benchmark", fake_synthetic_run)  # type: ignore[arg-type]
        monkeypatch.setattr(smoke_gate_module, "evaluate_longmemeval_recall", fake_lme_eval)  # type: ignore[arg-type]
        rc = smoke_gate_module.main(["--include-longmemeval"])  # type: ignore[attr-defined]
        assert rc == 0

    def test_main_with_include_longmemeval_fails_when_recall_breaches_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        """``--include-longmemeval`` exits 1 when the probe reports a sub-floor recall."""

        def fake_synthetic_run() -> tuple[int, dict[str, object]]:
            return 0, PAYLOAD_ALL_PASS

        def fake_lme_eval() -> tuple[bool, list[str]]:
            return False, ["LongMemEval recall@5 0.123 < floor 0.300"]

        monkeypatch.setattr(smoke_gate_module, "run_synthetic_benchmark", fake_synthetic_run)  # type: ignore[arg-type]
        monkeypatch.setattr(smoke_gate_module, "evaluate_longmemeval_recall", fake_lme_eval)  # type: ignore[arg-type]
        rc = smoke_gate_module.main(["--include-longmemeval"])  # type: ignore[attr-defined]
        assert rc == 1


class TestSubprocessIntegration:
    """The synthetic-runner subprocess hook handles its protocol correctly.

    We do not run the real ~5-minute benchmark here; instead the test
    stubs ``subprocess.run`` so the hook receives a synthetic payload
    written to the temporary output path and exits with code 0.
    """

    def test_run_synthetic_benchmark_parses_json_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        captured_args: dict[str, list[str]] = {}

        def fake_subprocess_run(
            command: list[str],
            *_args: object,
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            captured_args["command"] = command
            out_path_idx = command.index("--output-path") + 1
            Path(command[out_path_idx]).write_text(
                json.dumps(PAYLOAD_ALL_PASS),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
        exit_code, payload = smoke_gate_module.run_synthetic_benchmark()  # type: ignore[attr-defined]
        assert exit_code == 0
        assert payload == PAYLOAD_ALL_PASS
        assert captured_args["command"][0] == sys.executable
        assert "--output-format" in captured_args["command"]
        assert "json" in captured_args["command"]

    def test_run_synthetic_benchmark_forwards_missing_extra_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        smoke_gate_module: object,
    ) -> None:
        def fake_subprocess_run(
            command: list[str],
            *_args: object,
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                command,
                2,
                stdout="",
                stderr="missing sentence_transformers\n",
            )

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
        exit_code, payload = smoke_gate_module.run_synthetic_benchmark()  # type: ignore[attr-defined]
        assert exit_code == 2
        assert payload == {}
