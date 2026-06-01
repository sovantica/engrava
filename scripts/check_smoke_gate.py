"""Pre-publish quality gate for the public engrava release path.

Drives the bundled synthetic benchmark in its binding acceptance-
criterion (binding-AC) mode and asserts the four floors the release
freezes: synthesis coverage, direct-query neutrality, sanity-scenario
neutrality, and sanity-with-reflection-boost neutrality. An optional
LongMemEval real-data probe is invoked via ``--include-longmemeval``
and enforces a calibrated absolute recall@5 floor; the probe stays OFF
in CI by default because the dataset is user-download and the run takes
several minutes, but maintainers run it manually before tagging a
release.

Exit codes:
* ``0`` — every enforced floor met.
* ``1`` — at least one floor breached, the benchmark itself reported a
  failure, or LongMemEval was requested without a committed floor.
* ``2`` — the embeddings extras are missing (the synthetic runner uses
  this code for the same condition; the gate forwards it untouched).

The floors live in this file deliberately. They mirror the binding
values the benchmark already enforces (a second guard against the
runner's thresholds being relaxed silently). Raising them is allowed by
PR; lowering requires a governance amendment with empirical
justification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# Public floors. Mirror the frozen acceptance criteria committed to
# ``engrava/benchmarks/synthetic/runner.py`` — a deliberate redundancy so
# the gate fails if either side drifts. The four labels exactly match
# the benchmark's ``_BindingResult.label`` strings.
SYNTHETIC_FLOORS: Mapping[str, dict[str, float | str]] = {
    "AC-9a synthesis coverage": {"comparator": ">=", "floor": 0.80},
    "AC-9b direct neutrality": {"comparator": "<=", "floor": 0.05},
    "AC-8  sanity tolerance": {"comparator": "<=", "floor": 0.05},
    "AC-8b sanity (boost=1.0)": {"comparator": "<=", "floor": 0.05},
}

# LongMemEval recall@5 floor — calibrated against ``release/v0.3.0``'s
# behaviour on the upstream LongMemEval oracle variant in substring mode
# (``top_k=5``, dreaming ON, encoder
# ``sentence-transformers/all-MiniLM-L6-v2``, the harness default that
# ``resolve_embedding_provider_or_exit`` returns). The aggregate score
# the harness reported under those conditions was ``0.384`` on 498/500
# questions (two questions excluded — see
# ``LONGMEMEVAL_EXCLUDED_QUESTION_IDS``). The floor sits well below that
# aggregate to absorb the natural variation the dependent stack
# introduces between releases (encoder version, torch float ordering,
# SQLite tokenizer) without flagging non-regressions. Raising the floor
# in a future release is allowed by PR; lowering it requires governance
# approval and empirical justification.
LONGMEMEVAL_RECALL_AT_5_FLOOR: float | None = 0.30

# Questions that the upstream LongMemEval oracle variant ships with text
# that the engrava-core FTS5 query normaliser cannot serialise into a
# valid ``MATCH`` expression. The harness crashes mid-sweep when it
# reaches one of them, so the probe filters them out by id. Each entry
# corresponds to a tracked follow-up against the FTS normaliser; the set
# shrinks to empty once the normaliser handles single-quoted phrases
# plus a bare ``Not`` token in the same query without producing invalid
# FTS5 syntax.
LONGMEMEVAL_EXCLUDED_QUESTION_IDS: frozenset[str] = frozenset(
    {
        "352ab8bd",
        "gpt4_59149c77",
    }
)

SYNTHETIC_BENCHMARK_MODULE = "engrava.benchmarks.synthetic"
LONGMEMEVAL_DEFAULT_TOP_K = 5
LONGMEMEVAL_EVAL_MODE: Literal["substring", "cosine", "llm"] = "substring"
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_MISSING_EXTRA = 2


# ---------------------------------------------------------------------
# Synthetic gate
# ---------------------------------------------------------------------


def run_synthetic_benchmark(
    *,
    python_executable: str | None = None,
) -> tuple[int, dict[str, object]]:
    """Run the binding-AC synthetic benchmark and return ``(exit_code, payload)``.

    The benchmark is invoked through its public ``python -m`` entry
    point — the same surface the Makefile and any operator would use.
    Output is requested in JSON form into a temporary file so the
    parser does not race with the runner's stdout writes.
    """
    python_executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="engrava-smoke-gate-") as tmp_dir:
        out_path = Path(tmp_dir) / "binding.json"
        command = [
            python_executable,
            "-m",
            SYNTHETIC_BENCHMARK_MODULE,
            "--output-format",
            "json",
            "--output-path",
            str(out_path),
        ]
        completed = subprocess.run(  # noqa: S603 — trusted internal CLI invocation
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == EXIT_MISSING_EXTRA:
            sys.stderr.write(completed.stderr)
            return EXIT_MISSING_EXTRA, {}
        if not out_path.exists():
            sys.stderr.write(
                "synthetic benchmark did not produce the requested JSON output:\n"
                f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}\n",
            )
            return EXIT_FAIL, {}
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sys.stderr.write(
                f"synthetic benchmark produced malformed JSON: {exc}\n",
            )
            return EXIT_FAIL, {}
        return completed.returncode, payload


def _compare(comparator: str, numeric_value: float, floor: float) -> str | None:
    """Return a ``"too low / too high"`` clause when ``numeric_value`` violates the floor."""
    if comparator == ">=":
        if numeric_value < floor:
            return f"{numeric_value:.3f} < floor {floor:.2f} (must be >= {floor:.2f})"
        return None
    if comparator == "<=":
        if numeric_value > floor:
            return f"{numeric_value:.3f} > ceiling {floor:.2f} (must be <= {floor:.2f})"
        return None
    return f"unsupported comparator {comparator!r} in SYNTHETIC_FLOORS"


def _check_one_floor(
    label: str,
    rule: Mapping[str, float | str],
    entry: Mapping[str, object] | None,
) -> str | None:
    """Return a failure message for one floor, or ``None`` if the floor holds."""
    if entry is None:
        return f"synthetic benchmark JSON missing binding result for {label!r}"
    value = entry.get("value")
    if not isinstance(value, (int, float)):
        return f"{label}: benchmark JSON value is not numeric (got {value!r})"
    clause = _compare(str(rule["comparator"]), float(value), float(rule["floor"]))
    if clause is None:
        return None
    return f"{label} {clause}"


def evaluate_synthetic_floors(
    payload: Mapping[str, object],
) -> tuple[bool, list[str]]:
    """Apply :data:`SYNTHETIC_FLOORS` to the benchmark JSON payload."""
    binding_section = payload.get("binding")
    if not isinstance(binding_section, list):
        return False, ["synthetic benchmark JSON missing the 'binding' section"]

    by_label: dict[str, Mapping[str, object]] = {}
    for entry in binding_section:
        if isinstance(entry, dict) and isinstance(entry.get("label"), str):
            by_label[entry["label"]] = entry

    failures: list[str] = []
    for label, rule in SYNTHETIC_FLOORS.items():
        failure = _check_one_floor(label, rule, by_label.get(label))
        if failure is not None:
            failures.append(failure)

    return not failures, failures


# ---------------------------------------------------------------------
# LongMemEval probe (optional)
# ---------------------------------------------------------------------


def _check_recall_against_floor(
    recall_at_5: float,
    *,
    floor: float,
) -> tuple[bool, list[str]]:
    """Compare an observed recall@5 number against the committed floor."""
    if recall_at_5 < floor:
        return False, [
            f"LongMemEval recall@{LONGMEMEVAL_DEFAULT_TOP_K} {recall_at_5:.3f} < floor {floor:.3f}",
        ]
    return True, []


def evaluate_longmemeval_recall(
    *,
    floor: float | None = LONGMEMEVAL_RECALL_AT_5_FLOOR,
) -> tuple[bool, list[str]]:
    """Run the LongMemEval substring probe at ``top_k=5`` and check the floor.

    The probe imports the runner directly because the LongMemEval CLI
    does not emit machine-readable output. ``aggregate_score`` at
    ``top_k=5`` in substring mode is the recall@5 number. Questions in
    :data:`LONGMEMEVAL_EXCLUDED_QUESTION_IDS` are filtered out before the
    run because the FTS5 query normaliser crashes on their text.
    """
    if floor is None:
        return False, [
            "LongMemEval was requested but the recall@5 floor is not yet "
            "calibrated; commit the empirical value before enabling the "
            "probe in CI",
        ]

    try:
        from engrava.benchmarks.longmemeval.dataset_loader import (  # noqa: PLC0415
            DatasetDownloadError,
            load_dataset,
        )
        from engrava.benchmarks.longmemeval.runner import (  # noqa: PLC0415
            run_longmemeval_sync,
        )
        from engrava.benchmarks.synthetic.evaluate import (  # noqa: PLC0415
            resolve_embedding_provider_or_exit,
        )
    except ImportError as exc:
        return False, [f"LongMemEval imports unavailable: {exc}"]

    try:
        all_questions = load_dataset(variant="oracle")
    except DatasetDownloadError as exc:
        return False, [f"LongMemEval dataset unavailable: {exc}"]

    questions = [
        question
        for question in all_questions
        if question.question_id not in LONGMEMEVAL_EXCLUDED_QUESTION_IDS
    ]
    if not questions:
        return False, [
            "LongMemEval dataset filtered to zero questions — every entry "
            "matched the excluded id list, which should never happen on the "
            "upstream oracle variant",
        ]

    provider = resolve_embedding_provider_or_exit()
    results = run_longmemeval_sync(
        questions,
        dreaming_enabled=True,
        embedding_provider=provider,
        eval_mode=LONGMEMEVAL_EVAL_MODE,
        retrieval_top_k=LONGMEMEVAL_DEFAULT_TOP_K,
    )
    # Operator-facing disclosure: the gate must announce both how many
    # questions actually ran AND how many it filtered out by id, so a
    # silent shrinkage of the dataset cannot pass unnoticed.
    sys.stdout.write(
        f"LongMemEval: evaluated {results.total_questions}, "
        f"excluded {len(LONGMEMEVAL_EXCLUDED_QUESTION_IDS)} known FTS-normalizer cases\n",
    )
    sys.stdout.write(
        f"LongMemEval observed recall@{LONGMEMEVAL_DEFAULT_TOP_K}: "
        f"{float(results.aggregate_score):.3f} (floor {floor:.2f})\n",
    )
    return _check_recall_against_floor(float(results.aggregate_score), floor=floor)


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


def _format_floor_table() -> str:
    rows = []
    for label, rule in SYNTHETIC_FLOORS.items():
        rows.append(f"  {label:<32}  {rule['comparator']} {float(rule['floor']):.2f}")
    return "\n".join(rows)


def _print_synthetic_report(
    payload: Mapping[str, object],
    *,
    passed: bool,
    failures: Sequence[str],
) -> None:
    sys.stdout.write("Synthetic binding-AC measurements\n")
    sys.stdout.write("-" * 60 + "\n")
    binding_section = payload.get("binding")
    if isinstance(binding_section, list):
        for entry in binding_section:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "?"))
            value = entry.get("value", "?")
            value_str = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            rule_text = str(entry.get("rule_text", "?"))
            entry_passed = entry.get("passed", False)
            verdict = "PASS" if entry_passed else "FAIL"
            sys.stdout.write(f"  {label:<32}({rule_text}):  {value_str:>6}   {verdict}\n")
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write(f"Synthetic gate: {'PASS' if passed else 'FAIL'}\n")
    for failure in failures:
        sys.stdout.write(f"  - {failure}\n")
    sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the pre-publish quality gate."""
    parser = argparse.ArgumentParser(
        prog="check_smoke_gate.py",
        description=(
            "Pre-publish quality gate. Runs the synthetic benchmark in "
            "binding-AC mode and (optionally) the LongMemEval recall "
            "probe; exits non-zero if any committed floor is breached."
        ),
    )
    parser.add_argument(
        "--include-longmemeval",
        action="store_true",
        help=(
            "Also run the LongMemEval substring probe at top_k=5. Off in "
            "CI by default; the recall@5 floor must be calibrated and "
            "committed before this flag is wired into a release path."
        ),
    )
    args = parser.parse_args(argv)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Engrava pre-publish smoke gate\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Committed synthetic floors:\n")
    sys.stdout.write(_format_floor_table() + "\n")
    if LONGMEMEVAL_RECALL_AT_5_FLOOR is not None:
        sys.stdout.write(
            f"Committed LongMemEval recall@{LONGMEMEVAL_DEFAULT_TOP_K} floor: "
            f">= {LONGMEMEVAL_RECALL_AT_5_FLOOR:.2f}\n",
        )
    sys.stdout.write("\n")

    syn_exit, syn_payload = run_synthetic_benchmark()
    if syn_exit == EXIT_MISSING_EXTRA:
        sys.stdout.write(
            "Embeddings extras unavailable — gate cannot run. Install "
            "with: pip install 'engrava[embeddings-local]'\n",
        )
        return EXIT_MISSING_EXTRA

    syn_passed, syn_failures = evaluate_synthetic_floors(syn_payload)
    if syn_exit != EXIT_OK:
        syn_passed = False
        syn_failures = [
            *syn_failures,
            (
                "synthetic benchmark itself returned a non-zero exit "
                f"({syn_exit}) — at least one of its own AC checks failed"
            ),
        ]
    _print_synthetic_report(syn_payload, passed=syn_passed, failures=syn_failures)

    lme_passed = True
    lme_failures: list[str] = []
    if args.include_longmemeval:
        sys.stdout.write("LongMemEval real-data probe (substring, top_k=5)\n")
        sys.stdout.write("-" * 60 + "\n")
        lme_passed, lme_failures = evaluate_longmemeval_recall()
        sys.stdout.write(f"LongMemEval gate: {'PASS' if lme_passed else 'FAIL'}\n")
        for failure in lme_failures:
            sys.stdout.write(f"  - {failure}\n")
        sys.stdout.write("\n")

    overall_passed = syn_passed and lme_passed
    sys.stdout.write("=" * 60 + "\n")
    if overall_passed:
        sys.stdout.write("SMOKE GATE: PASS\n")
    else:
        sys.stdout.write("SMOKE GATE: FAIL — publish blocked\n")
        sys.stdout.write(
            "Human review required. Floor changes need empirical "
            "justification per the release governance.\n",
        )
    sys.stdout.write("=" * 60 + "\n")

    return EXIT_OK if overall_passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
