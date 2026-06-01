"""Command-line front-end for the LongMemEval Free harness.

Invoke as ``python -m engrava.benchmarks.longmemeval``. The default
configuration runs the ``oracle`` variant in substring-evaluation mode
without any LLM dependency. Flags expose the cosine and LLM modes, the
variant choice, dreaming OFF/ON arms, and an optional subset filter on
``question_type``.

Exit codes:

* ``0`` — at least one question was evaluated, no fatal error.
* ``1`` — a runtime error fired during the run (download, parsing,
  unsupported configuration).
* ``2`` — the embeddings extras are missing (re-uses the synthetic
  benchmark's clean-exit convention).
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from engrava.benchmarks.longmemeval.dataset_loader import (
    DatasetDownloadError,
    LongMemEvalQuestion,
    load_dataset,
)
from engrava.benchmarks.longmemeval.runner import (
    DEFAULT_TOP_K,
    EvalMode,
    LongMemEvalResults,
    run_longmemeval_sync,
)
from engrava.benchmarks.synthetic.evaluate import (
    resolve_embedding_provider_or_exit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]


_VALID_VARIANTS = ("oracle", "s", "m")
# CLI offers the two deterministic Free-side modes only. The LLM-judge
# mode requires a caller-supplied client implementing
# :class:`engrava.benchmarks.longmemeval.evaluate.LLMJudgeClient` and is
# accessible by driving :func:`run_longmemeval` from Python — see the
# README for a code snippet.
_CLI_MODES: tuple[EvalMode, ...] = ("substring", "cosine")


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the LongMemEval harness from command-line arguments.

    Args:
        argv: Argument vector; defaults to :data:`sys.argv` when ``None``.

    Returns:
        Process exit code (see module docstring).

    """
    args = _parse_args(argv)
    try:
        questions = load_dataset(variant=args.variant)
    except DatasetDownloadError as exc:
        sys.stderr.write(f"LongMemEval: {exc}\n")
        return 1

    if args.subset:
        questions = [q for q in questions if q.question_type == args.subset]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        sys.stderr.write(
            "LongMemEval: no questions to evaluate after applying subset/limit filters.\n",
        )
        return 1

    provider = resolve_embedding_provider_or_exit()
    try:
        results = run_longmemeval_sync(
            questions,
            dreaming_enabled=args.dreaming,
            embedding_provider=provider,
            eval_mode=args.eval_mode,
            retrieval_top_k=args.top_k,
        )
    except DatasetDownloadError as exc:
        sys.stderr.write(f"LongMemEval: {exc}\n")
        return 1

    _emit_report(results, questions=questions)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m engrava.benchmarks.longmemeval",
        description=(
            "Run the LongMemEval public benchmark against engrava. The "
            "default invocation uses the oracle variant in substring "
            "mode (no LLM required, deterministic)."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=_VALID_VARIANTS,
        default="oracle",
        help="Dataset variant to load (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=_CLI_MODES,
        default="substring",
        help=(
            "Evaluation mode (default: %(default)s). The LLM-judge mode "
            "is reachable only from Python; see README.md."
        ),
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Filter questions by question_type (e.g. single-session-recall). "
            "Omit to evaluate every question in the loaded variant."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions evaluated (default: no cap).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Retrieval top-K passed to search_hybrid (default: %(default)s).",
    )
    parser.add_argument(
        "--dreaming",
        dest="dreaming",
        action="store_true",
        help="Run one dreaming consolidation cycle per question.",
    )
    parser.add_argument(
        "--no-dreaming",
        dest="dreaming",
        action="store_false",
        help="Disable dreaming for this run (default).",
    )
    parser.set_defaults(dreaming=False)
    return parser.parse_args(argv)


def _emit_report(
    results: LongMemEvalResults,
    *,
    questions: list[LongMemEvalQuestion],
) -> None:
    """Print a human-readable summary of the run."""
    sys.stdout.write("LongMemEval Free harness\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(f"variant questions evaluated: {results.total_questions}\n")
    sys.stdout.write(f"dreaming enabled: {results.dreaming_enabled}\n")
    sys.stdout.write(f"top_k: {results.top_k}\n")
    sys.stdout.write(f"eval_mode: {results.eval_mode}\n")
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write(f"aggregate score : {results.aggregate_score:.4f}\n")
    sys.stdout.write(f"raw signal mean : {results.aggregate_raw_signal:.4f}\n")
    if results.per_type:
        sys.stdout.write("\nper question_type:\n")
        for qtype, score in sorted(results.per_type.items()):
            count = sum(1 for q in questions if q.question_type == qtype)
            sys.stdout.write(f"  {qtype:<32} ({count:>3}): {score:.4f}\n")
    sys.stdout.write("=" * 60 + "\n")
