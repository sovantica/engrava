"""LongMemEval Free harness — public benchmark runner for engrava.

Uses the publicly distributed LongMemEval dataset (Wu et al., ICLR 2025,
arXiv:2410.10813). The dataset is **not bundled** with engrava; users
download it from the upstream HuggingFace distribution at runtime via
``load_dataset``.

The CLI exposes two deterministic evaluation modes (``substring`` and
``cosine``); both run without LLM API keys. LLM-judge mode is opt-in
and reachable only by driving ``run_longmemeval`` from Python with a
caller-supplied ``LLMJudgeClient`` — see
``benchmarks/longmemeval/README.md`` for a code snippet, attribution,
license, and download instructions.
"""

from __future__ import annotations

from engrava.benchmarks.longmemeval.dataset_loader import (
    LongMemEvalQuestion,
    LongMemEvalSession,
    LongMemEvalTurn,
    load_dataset,
)

__all__ = [
    "LongMemEvalQuestion",
    "LongMemEvalSession",
    "LongMemEvalTurn",
    "load_dataset",
]
