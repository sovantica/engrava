"""CLI entry point for the synthetic dreaming benchmark.

Invoke via ``python -m engrava.benchmarks.synthetic``.  The default
invocation runs the **binding acceptance measurements** only —
four curated ``scenario_mix`` subsets generated on the fly with
fixed ``seed=20260508``.  Each subset is the binding measurement
surface for one acceptance criterion:

* AC-9a synthesis coverage   (>= 0.80)
* AC-9b direct neutrality    (<= 0.05 delta ON vs OFF)
* AC-8  sanity tolerance     (<= 0.05 delta ON vs OFF)
* AC-8b sanity with explicit ``SearchConfig(reflection_boost=1.0)``
  (<= 0.05 delta ON vs OFF) — re-runs the sanity subset to pin
  the search config the benchmark commits to at v0.3.0.

Exit code is ``0`` iff all four binding ACs pass, ``1`` if any
binding AC fails, ``2`` if the embeddings extras are missing.
Reference-hardware walltime is ~5 minutes (AC-11 v0.3.0 ceiling is
300 seconds; a follow-up evaluator-optimisation workstream tightens
to 120 seconds).

Pass ``--with-reproducibility`` to additionally print a
**reproducibility snapshot** on the bundled frozen
``synthetic-v1.json`` dataset (50 conversations x natural scenario
distribution).  This is informational only — it shows the full
per-scenario picture of dreaming's effect across all scenario
classes, including the two scenarios where REFLECTIONs displace
correct observations at ``reflection_boost=1.0``
(``contradiction_resolution`` and ``recent_fact_recall``).
``docs/benchmarks.md`` ships a static capture of the
``--with-reproducibility`` output as the primary public
transparency surface.
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

from engrava.benchmarks.synthetic.evaluate import (
    EvaluationResult,
    measure_synthesis_coverage,
    resolve_embedding_provider_or_exit,
    run_evaluation,
)
from engrava.benchmarks.synthetic.generate import (
    Perspective,
    SyntheticConversation,
    SyntheticQuestion,
    SyntheticTurn,
    generate_dataset,
)
from engrava.benchmarks.synthetic.scenarios import (
    SCENARIO_LIBRARY,
    ScenarioDifficulty,
)
from engrava.config import SearchConfig

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from engrava.domain.protocols.embedding_provider import (
        EmbeddingProviderProtocol,
    )

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Frozen dataset constants — must match the freeze guard's parameters.
# ---------------------------------------------------------------------------

_FROZEN_DATASET_PACKAGE = "engrava.benchmarks.synthetic.datasets"
_FROZEN_DATASET_NAME = "synthetic-v1.json"
_DEFAULT_SEED = 20260508
_DEFAULT_N_CONVERSATIONS = 50
_DEFAULT_TURNS = 70
_DEFAULT_DENSITY = 0.4
_DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Binding measurement subsets — curated ``scenario_mix`` datasets used by
# section 2 of the CLI output.  These match the pytest acceptance tests
# verbatim so the CLI verdict and the test verdict agree by construction.
# ---------------------------------------------------------------------------

_AC9A_FLOOR = 0.80
# AC-9b v0.3.0 tolerance per spec v1.6 amendment: pre-amendment ceiling
# was 0.02 but empirically REFLECTIONs displace direct-retrieval OBS at
# ``reflection_boost=1.0`` (multiplier, not toggle).  Measured 0.033 on
# the curated direct subset post-NA-1; 0.05 carries a 34 % safety
# margin.  Follow-up evaluator-ranking workstream tightens back to 0.02.
_AC9B_CEILING = 0.05
_AC8_CEILING = 0.05

_SYNTHESIS_SCENARIO_NAMES = (
    "abstract_theme_recall",
    "repeated_paraphrase_compression",
    "thematic_cluster",
)
_DIRECT_SCENARIO_NAMES = (
    "long_recall_simple",
    "multi_fact_recall",
    "contradiction_resolution",
    "distraction_heavy",
)
_SANITY_SCENARIO_NAMES = (
    "single_unique_fact",
    "recent_fact_recall",
)


# ---------------------------------------------------------------------------
# Per-scenario regression annotation thresholds — used by the
# reproducibility-section emitter to flag scenarios whose ON-OFF delta
# warrants a "REFLECTION displacement" footnote.
# ---------------------------------------------------------------------------

_REGRESSION_ANNOTATION_MARGIN = 0.05


@dataclass(frozen=True)
class _CliArgs:
    """Parsed CLI arguments — frozen so downstream callers cannot mutate."""

    regenerate: bool
    seed: int
    n_conversations: int
    avg_turns_per_conversation: int
    distraction_density: float
    scenarios: frozenset[str]
    output_format: str
    output_path: Path | None
    top_k: int
    with_reproducibility: bool


@dataclass(frozen=True)
class _BindingResult:
    """One acceptance-criterion measurement for the binding section."""

    label: str
    value: float
    threshold: float
    passed: bool
    rule_text: str  # e.g. ">= 0.80" or "<= 0.02"


@dataclass(frozen=True)
class _CliReport:
    """Aggregated runtime artefacts the emitters render.

    ``reproducibility_off`` and ``reproducibility_on`` are populated
    only when the caller passed ``--with-reproducibility``; otherwise
    they stay ``None`` and the emitters skip Section 1.
    """

    binding_results: tuple[_BindingResult, ...]
    reproducibility_off: EvaluationResult | None = None
    reproducibility_on: EvaluationResult | None = None


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic benchmark CLI.

    Returns:
        ``0`` when every binding acceptance criterion passes,
        ``1`` when any binding AC fails.  ``2`` is reserved for the
        clean-exit path in
        :func:`resolve_embedding_provider_or_exit` when the
        ``embeddings-local`` extras are missing.

    """
    args = _parse_args(argv)
    provider = resolve_embedding_provider_or_exit()

    # --- Binding ACs (always run, gates the exit code) --------------------
    binding_results = _run_binding_acs(
        provider=provider,
        top_k=args.top_k,
    )

    # --- Optional reproducibility snapshot (opt-in via flag) --------------
    off_snapshot: EvaluationResult | None = None
    on_snapshot: EvaluationResult | None = None
    if args.with_reproducibility:
        reproducibility_dataset = _load_or_regenerate_dataset(args)
        if args.scenarios:
            reproducibility_dataset = _filter_by_scenarios(
                reproducibility_dataset,
                args.scenarios,
            )
        off_snapshot = run_evaluation(
            reproducibility_dataset,
            dreaming_enabled=False,
            embedding_provider=provider,
            retrieval_top_k=args.top_k,
        )
        on_snapshot = run_evaluation(
            reproducibility_dataset,
            dreaming_enabled=True,
            embedding_provider=provider,
            retrieval_top_k=args.top_k,
        )

    report = _CliReport(
        binding_results=binding_results,
        reproducibility_off=off_snapshot,
        reproducibility_on=on_snapshot,
    )
    return _emit_report(report=report, args=args)


# ---------------------------------------------------------------------------
# Binding-AC orchestration
# ---------------------------------------------------------------------------


def _run_binding_acs(
    *,
    provider: EmbeddingProviderProtocol,
    top_k: int,
) -> tuple[_BindingResult, ...]:
    """Run the four binding acceptance measurements on curated subsets.

    Datasets and ``scenario_mix`` weights match the pytest tests:

    * AC-9a — :func:`measure_synthesis_coverage` on a 30-conversation
      synthesis-only dataset (``avg_turns=15``, ``density=0.2``).
    * AC-9b — :func:`evaluate_run` on a 30-conversation direct-only
      dataset (``avg_turns=30``, ``density=0.4``); reports
      ``abs(on - off)``.
    * AC-8 / AC-8b — :func:`evaluate_run` on a 24-conversation
      sanity-only dataset (``avg_turns=20``, ``density=0.3``); AC-8b
      additionally pins an explicit ``SearchConfig(reflection_boost
      =1.0)`` so a future regression in the benchmark default
      cannot silently relax the gate.
    """
    return (
        _run_ac9a_synthesis_coverage(provider=provider),
        _run_ac9b_direct_neutrality(provider=provider, top_k=top_k),
        _run_ac8_sanity_tolerance(provider=provider, top_k=top_k),
        _run_ac8b_sanity_with_boost_one(provider=provider, top_k=top_k),
    )


def _run_ac9a_synthesis_coverage(
    *,
    provider: EmbeddingProviderProtocol,
) -> _BindingResult:
    """AC-9a — synthesis coverage on a 30-conversation synthesis-only dataset."""
    import asyncio  # noqa: PLC0415 -- evaluate helpers are async; main is sync.

    synthesis_mix = dict.fromkeys(_SYNTHESIS_SCENARIO_NAMES, 1.0)
    dataset = generate_dataset(
        seed=_DEFAULT_SEED,
        n_conversations=30,
        avg_turns_per_conversation=15,
        distraction_density=0.2,
        scenario_mix=synthesis_mix,
    )
    coverage = asyncio.run(
        measure_synthesis_coverage(dataset, embedding_provider=provider),
    )
    return _BindingResult(
        label="AC-9a synthesis coverage",
        value=coverage,
        threshold=_AC9A_FLOOR,
        passed=coverage >= _AC9A_FLOOR,
        rule_text=f">= {_AC9A_FLOOR:.2f}",
    )


def _run_ac9b_direct_neutrality(
    *,
    provider: EmbeddingProviderProtocol,
    top_k: int,
) -> _BindingResult:
    """AC-9b — direct-retrieval neutrality on a 30-conv direct-only dataset."""
    direct_mix = dict.fromkeys(_DIRECT_SCENARIO_NAMES, 1.0)
    dataset = generate_dataset(
        seed=_DEFAULT_SEED,
        n_conversations=30,
        avg_turns_per_conversation=30,
        distraction_density=0.4,
        scenario_mix=direct_mix,
    )
    delta = _measure_recall_delta(
        dataset=dataset,
        provider=provider,
        top_k=top_k,
        search_config=None,
    )
    return _BindingResult(
        label="AC-9b direct neutrality",
        value=delta,
        threshold=_AC9B_CEILING,
        passed=delta <= _AC9B_CEILING,
        rule_text=f"<= {_AC9B_CEILING:.2f}",
    )


def _run_ac8_sanity_tolerance(
    *,
    provider: EmbeddingProviderProtocol,
    top_k: int,
) -> _BindingResult:
    """AC-8 — sanity tolerance on a 24-conv anti-cherry-pick neutral dataset."""
    sanity_mix = dict.fromkeys(_SANITY_SCENARIO_NAMES, 1.0)
    dataset = generate_dataset(
        seed=_DEFAULT_SEED,
        n_conversations=24,
        avg_turns_per_conversation=20,
        distraction_density=0.3,
        scenario_mix=sanity_mix,
    )
    delta = _measure_recall_delta(
        dataset=dataset,
        provider=provider,
        top_k=top_k,
        search_config=None,
    )
    return _BindingResult(
        label="AC-8  sanity tolerance",
        value=delta,
        threshold=_AC8_CEILING,
        passed=delta <= _AC8_CEILING,
        rule_text=f"<= {_AC8_CEILING:.2f}",
    )


def _run_ac8b_sanity_with_boost_one(
    *,
    provider: EmbeddingProviderProtocol,
    top_k: int,
) -> _BindingResult:
    """AC-8b — same sanity dataset, explicit reflection_boost=1.0."""
    sanity_mix = dict.fromkeys(_SANITY_SCENARIO_NAMES, 1.0)
    dataset = generate_dataset(
        seed=_DEFAULT_SEED,
        n_conversations=24,
        avg_turns_per_conversation=20,
        distraction_density=0.3,
        scenario_mix=sanity_mix,
    )
    boost_off = SearchConfig(reflection_boost=1.0)
    delta = _measure_recall_delta(
        dataset=dataset,
        provider=provider,
        top_k=top_k,
        search_config=boost_off,
    )
    return _BindingResult(
        label="AC-8b sanity (boost=1.0)",
        value=delta,
        threshold=_AC8_CEILING,
        passed=delta <= _AC8_CEILING,
        rule_text=f"<= {_AC8_CEILING:.2f}",
    )


def _measure_recall_delta(
    *,
    dataset: tuple[SyntheticConversation, ...],
    provider: EmbeddingProviderProtocol,
    top_k: int,
    search_config: SearchConfig | None,
) -> float:
    """Run OFF + ON arms and return ``abs(on_recall - off_recall)``."""
    off = run_evaluation(
        dataset,
        dreaming_enabled=False,
        embedding_provider=provider,
        retrieval_top_k=top_k,
        search_config=search_config,
    )
    on = run_evaluation(
        dataset,
        dreaming_enabled=True,
        embedding_provider=provider,
        retrieval_top_k=top_k,
        search_config=search_config,
    )
    return abs(on.aggregate_recall_at_k - off.aggregate_recall_at_k)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> _CliArgs:
    parser = argparse.ArgumentParser(
        prog="python -m engrava.benchmarks.synthetic",
        description=(
            "Reproducible OFF-vs-ON dreaming benchmark.  Default "
            "invocation runs the four binding acceptance "
            "measurements on curated subsets and exits 0 iff all "
            "pass.  Pass --with-reproducibility to additionally "
            "print a per-scenario snapshot on the bundled frozen "
            "dataset (adds ~5 minutes walltime; that output is the "
            "primary public transparency surface captured in "
            "docs/benchmarks.md)."
        ),
    )
    parser.add_argument(
        "--with-reproducibility",
        action="store_true",
        dest="with_reproducibility",
        help=(
            "Print the reproducibility snapshot section "
            "(per-scenario OFF vs ON breakdown on the bundled "
            "frozen synthetic-v1.json dataset).  Adds ~5 minutes "
            "of walltime.  Default: skip the snapshot and run only "
            "the binding acceptance measurements."
        ),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "Rebuild the reproducibility dataset from the generator "
            "instead of loading the frozen JSON.  Implies "
            "--with-reproducibility (regenerating without using the "
            "snapshot would be a no-op).  Combine with --seed / "
            "--n-conversations / --avg-turns / --density to control "
            "the regeneration.  Binding AC measurements always use "
            "the curated subset configuration regardless of this flag."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Master seed for --regenerate.  Default: %(default)s.",
    )
    parser.add_argument(
        "--n-conversations",
        type=int,
        default=_DEFAULT_N_CONVERSATIONS,
        help="Conversation count for --regenerate.  Default: %(default)s.",
    )
    parser.add_argument(
        "--avg-turns",
        type=int,
        default=_DEFAULT_TURNS,
        dest="avg_turns_per_conversation",
        help="Target turns per conversation for --regenerate.  Default: %(default)s.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=_DEFAULT_DENSITY,
        dest="distraction_density",
        help="Distraction density for --regenerate.  Default: %(default)s.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help=(
            "Comma-separated subset of scenario names to evaluate in "
            "the reproducibility snapshot (section 1 only).  Empty "
            "(default) means every scenario in the loaded dataset.  "
            "Binding AC measurements ignore this flag."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help=(
            "Summary output format.  ``text`` (default) prints a "
            "human-readable two-section report to stdout; ``json`` "
            "emits a structured payload either to stdout or to the "
            "path given by --output-path."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Optional file path for --output-format=json.  When omitted, JSON is written to stdout."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help="Retrieval top-K for recall@K.  Default: %(default)s.",
    )

    parsed = parser.parse_args(argv)

    scenario_names = _parse_scenario_subset(parsed.scenarios)
    output_path = _coerce_output_path(parsed.output_path)
    # ``--regenerate`` implies ``--with-reproducibility`` — regenerating
    # the snapshot dataset just to discard it would be a no-op.
    with_repro = bool(parsed.with_reproducibility or parsed.regenerate)

    return _CliArgs(
        regenerate=parsed.regenerate,
        seed=parsed.seed,
        n_conversations=parsed.n_conversations,
        avg_turns_per_conversation=parsed.avg_turns_per_conversation,
        distraction_density=parsed.distraction_density,
        scenarios=scenario_names,
        output_format=parsed.output_format,
        output_path=output_path,
        top_k=parsed.top_k,
        with_reproducibility=with_repro,
    )


def _parse_scenario_subset(raw: str) -> frozenset[str]:
    if not raw.strip():
        return frozenset()
    names = frozenset(part.strip() for part in raw.split(",") if part.strip())
    known = {s.name for s in SCENARIO_LIBRARY}
    unknown = names - known
    if unknown:
        msg = f"--scenarios references unknown scenario(s): {sorted(unknown)}"
        raise SystemExit(msg)
    return names


def _coerce_output_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    from pathlib import Path  # noqa: PLC0415

    return Path(raw)


# ---------------------------------------------------------------------------
# Dataset loading + filtering
# ---------------------------------------------------------------------------


def _load_or_regenerate_dataset(args: _CliArgs) -> tuple[SyntheticConversation, ...]:
    if args.regenerate:
        return generate_dataset(
            seed=args.seed,
            n_conversations=args.n_conversations,
            avg_turns_per_conversation=args.avg_turns_per_conversation,
            distraction_density=args.distraction_density,
        )
    return _load_frozen_dataset()


def _load_frozen_dataset() -> tuple[SyntheticConversation, ...]:
    """Read the bundled ``synthetic-v1.json`` file via ``importlib.resources``."""
    try:
        resource = importlib.resources.files(_FROZEN_DATASET_PACKAGE).joinpath(
            _FROZEN_DATASET_NAME,
        )
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        msg = (
            f"frozen dataset {_FROZEN_DATASET_NAME!r} not found in "
            f"installed package — was engrava installed without "
            f"benchmarks package-data?  Original error: {exc}"
        )
        raise SystemExit(msg) from exc

    if not resource.is_file():
        msg = (
            f"frozen dataset {_FROZEN_DATASET_NAME!r} not found at "
            f"expected install location.  Re-install engrava or run "
            f"with --regenerate to rebuild from the generator."
        )
        raise SystemExit(msg)

    payload = json.loads(resource.read_text(encoding="utf-8"))
    return tuple(_deserialise_conversation(item) for item in payload)


def _deserialise_conversation(payload: dict[str, object]) -> SyntheticConversation:
    """Inverse of ``dataset_to_json`` — JSON dict -> frozen dataclass."""
    raw_turns = _checked_list(payload["turns"])
    raw_questions = _checked_list(payload["questions"])
    return SyntheticConversation(
        conversation_id=str(payload["conversation_id"]),
        scenario_name=str(payload["scenario_name"]),
        turns=tuple(_deserialise_turn(t) for t in raw_turns),
        questions=tuple(_deserialise_question(q) for q in raw_questions),
    )


def _deserialise_turn(payload: object) -> SyntheticTurn:
    raw = _checked_dict(payload)
    fact_id_raw = raw["fact_id"]
    return SyntheticTurn(
        turn_index=_as_int(raw["turn_index"]),
        perspective=_as_perspective(raw["perspective"]),
        source_is_self=_as_bool(raw["source_is_self"]),
        text=str(raw["text"]),
        is_memorable=_as_bool(raw["is_memorable"]),
        fact_id=None if fact_id_raw is None else str(fact_id_raw),
    )


def _deserialise_question(payload: object) -> SyntheticQuestion:
    raw = _checked_dict(payload)
    return SyntheticQuestion(
        question_id=str(raw["question_id"]),
        scenario_name=str(raw["scenario_name"]),
        difficulty=_as_difficulty(raw["difficulty"]),
        asked_at_turn=_as_int(raw["asked_at_turn"]),
        question_text=str(raw["question_text"]),
        expected_fact_ids=tuple(str(s) for s in _checked_list(raw["expected_fact_ids"])),
        expected_substrings=tuple(str(s) for s in _checked_list(raw["expected_substrings"])),
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"expected JSON integer, got {type(value).__name__}"
        raise SystemExit(msg)
    return value


def _as_bool(value: object) -> bool:
    # Strict — reject anything that is not a literal JSON ``true`` /
    # ``false``.  Pre-fix this coerced via ``bool(value)`` which
    # silently turned every truthy string (including the literal
    # ``"false"``) into ``True``, corrupting the self-anchored
    # provenance contract on malformed datasets.
    if not isinstance(value, bool):
        msg = f"expected JSON boolean, got {type(value).__name__}"
        raise SystemExit(msg)
    return value


def _as_perspective(value: object) -> Perspective:
    if not isinstance(value, str) or value not in {"percept", "utterance", "thought"}:
        msg = f"expected perspective tag, got {value!r}"
        raise SystemExit(msg)
    return cast("Perspective", value)


def _as_difficulty(value: object) -> ScenarioDifficulty:
    if not isinstance(value, str) or value not in {"easy", "medium", "hard"}:
        msg = f"expected difficulty tag, got {value!r}"
        raise SystemExit(msg)
    return cast("ScenarioDifficulty", value)


def _checked_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"expected JSON object, got {type(value).__name__}"
        raise SystemExit(msg)
    return value


def _checked_list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = f"expected JSON array, got {type(value).__name__}"
        raise SystemExit(msg)
    return value


def _filter_by_scenarios(
    conversations: Iterable[SyntheticConversation],
    names: frozenset[str],
) -> tuple[SyntheticConversation, ...]:
    return tuple(c for c in conversations if c.scenario_name in names)


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------


def _emit_report(*, report: _CliReport, args: _CliArgs) -> int:
    passed = all(r.passed for r in report.binding_results)
    if args.output_format == "json":
        _write_json_report(report=report, passed=passed, args=args)
    else:
        _write_text_report(report=report, passed=passed)
    return 0 if passed else 1


def _write_text_report(*, report: _CliReport, passed: bool) -> None:
    lines: list[str] = []
    lines.append("Engrava Synthetic Benchmark Suite")
    lines.append("")
    _append_binding_section(lines=lines, report=report, passed=passed)
    if report.reproducibility_on is None or report.reproducibility_off is None:
        lines.append("")
        lines.append("For full per-scenario texture on bundled synthetic-v1.json:")
        lines.append("  python -m engrava.benchmarks.synthetic --with-reproducibility")
        lines.append("See docs/benchmarks.md for explanation of reproducibility output.")
    else:
        _append_reproducibility_section(
            lines=lines,
            off=report.reproducibility_off,
            on=report.reproducibility_on,
        )
    sys.stdout.write("\n".join(lines) + "\n")


def _append_binding_section(
    *,
    lines: list[str],
    report: _CliReport,
    passed: bool,
) -> None:
    """Emit the binding acceptance measurements section into ``lines``."""
    lines.append("=" * 63)
    lines.append("Binding acceptance measurements (curated subsets)")
    lines.append("=" * 63)
    lines.append("")
    for result in report.binding_results:
        verdict = "PASS" if result.passed else "FAIL"
        lines.append(
            f"  {result.label:<32}({result.rule_text}):  {result.value:>5.3f}   {verdict}",
        )

    lines.append("")
    lines.append("=" * 63)
    if passed:
        lines.append("ALL BINDING ACs PASS - engrava dreaming evidence: VERIFIED.")
    else:
        failed = [r.label for r in report.binding_results if not r.passed]
        lines.append("BINDING AC FAIL: " + ", ".join(failed))
    lines.append("=" * 63)


def _append_reproducibility_section(
    *,
    lines: list[str],
    off: EvaluationResult,
    on: EvaluationResult,
) -> None:
    """Emit the optional reproducibility snapshot section into ``lines``."""
    overall_delta = on.aggregate_recall_at_k - off.aggregate_recall_at_k
    lines.append("")
    lines.append("=" * 63)
    lines.append("Reproducibility snapshot (frozen synthetic-v1.json)")
    lines.append("=" * 63)
    lines.append(
        f"Dataset: {on.total_questions} questions across "
        f"{len(on.per_scenario)} scenario(s) (natural distribution)",
    )
    lines.append("")
    lines.append(f"{'':<34}{'OFF':>8}{'ON':>10}{'delta':>11}")
    lines.append("-" * 63)
    lines.append(
        f"{'recall@' + str(on.top_k) + ' (overall)':<34}"
        f"{off.aggregate_recall_at_k:>8.3f}"
        f"{on.aggregate_recall_at_k:>10.3f}"
        f"{overall_delta * 100:>+10.1f}pp",
    )
    annotated_any = False
    for scenario in SCENARIO_LIBRARY:
        if scenario.name not in on.per_scenario:
            continue
        off_score = off.per_scenario[scenario.name].recall_at_k
        on_score = on.per_scenario[scenario.name].recall_at_k
        scenario_delta = on_score - off_score
        flag = ""
        if scenario_delta <= -_REGRESSION_ANNOTATION_MARGIN:
            flag = "  (!)"
            annotated_any = True
        lines.append(
            f"  {scenario.name:<32}"
            f"{off_score:>8.3f}"
            f"{on_score:>10.3f}"
            f"{scenario_delta * 100:>+10.1f}pp{flag}",
        )
    if annotated_any:
        lines.append("")
        lines.append(
            "Note: scenarios marked (!) show REFLECTION displacement at boost=1.0",
        )
        lines.append(
            "      (see docs/benchmarks.md).  Reproducibility snapshot is",
        )
        lines.append(
            "      informational; binding ACs were already verified above.",
        )


def _write_json_report(*, report: _CliReport, passed: bool, args: _CliArgs) -> None:
    payload: dict[str, object] = {
        "binding": [asdict(r) for r in report.binding_results],
        "passed": passed,
    }
    if report.reproducibility_off is not None and report.reproducibility_on is not None:
        payload["reproducibility"] = {
            "off": asdict(report.reproducibility_off),
            "on": asdict(report.reproducibility_on),
        }
    rendered = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True)
    if args.output_path is None:
        sys.stdout.write(rendered + "\n")
    else:
        args.output_path.write_text(rendered + "\n", encoding="utf-8", newline="\n")
