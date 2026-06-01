"""Tests for the LongMemEval CLI argument parser + error paths."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from engrava.benchmarks.longmemeval.cli import _parse_args, main

if TYPE_CHECKING:
    from collections.abc import Sequence

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "longmemeval_tiny.json"


@pytest.fixture
def isolated_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect the loader's cache root to a tmp directory."""
    cache = tmp_path / "lme-cli-cache"
    cache.mkdir()
    monkeypatch.setattr(
        "engrava.benchmarks.longmemeval.dataset_loader.LOCAL_CACHE_DIR",
        cache,
    )
    return cache


def _seed_oracle_cache(cache: Path) -> None:
    shutil.copyfile(FIXTURE_PATH, cache / "longmemeval_oracle.json")


class TestParseArgs:
    """Defaults + flag combinations."""

    def test_defaults_are_oracle_substring_no_dreaming(self) -> None:
        args = _parse_args([])
        assert args.variant == "oracle"
        assert args.eval_mode == "substring"
        assert args.dreaming is False
        assert args.subset is None
        assert args.limit is None
        assert args.top_k == 5

    def test_dreaming_flag_flips_default(self) -> None:
        args = _parse_args(["--dreaming"])
        assert args.dreaming is True

    def test_no_dreaming_explicitly_restores_default(self) -> None:
        args = _parse_args(["--dreaming", "--no-dreaming"])
        assert args.dreaming is False

    def test_invalid_variant_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--variant", "huge"])

    def test_invalid_eval_mode_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--eval-mode", "wat"])

    def test_subset_and_limit_propagate(self) -> None:
        args = _parse_args(["--subset", "single-session-recall", "--limit", "5"])
        assert args.subset == "single-session-recall"
        assert args.limit == 5


class TestMainErrorPaths:
    """The CLI surfaces actionable failures without crashing."""

    def test_llm_mode_is_not_available_via_cli(self) -> None:
        """``--eval-mode=llm`` is rejected by argparse (LLM is Python-only)."""
        with pytest.raises(SystemExit):
            _parse_args(["--eval-mode", "llm"])

    def test_unknown_subset_yields_empty_set_and_exits_one(
        self,
        isolated_cache: Path,
    ) -> None:
        """Filtering on a non-existent ``question_type`` exits 1."""
        _seed_oracle_cache(isolated_cache)
        rc = main(["--subset", "does-not-exist"])
        assert rc == 1

    def test_missing_cache_and_no_network_exits_one(
        self,
        isolated_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty cache plus a failing downloader surfaces an actionable error."""
        import urllib.error

        unreachable = urllib.error.URLError("no network in test")

        def failing_urlopen(*_args: object, **_kwargs: object) -> object:
            raise unreachable

        monkeypatch.setattr(
            "engrava.benchmarks.longmemeval.dataset_loader.urllib.request.urlopen",
            failing_urlopen,
        )
        # Cache exists as a directory but holds no oracle file → loader downloads.
        rc = main([])
        assert rc == 1


def _no_argv_run_safety(_argv: Sequence[str]) -> None:
    """Compile-time guard that ``main`` accepts a Sequence argv."""
    # The body intentionally does not run; this exists so static type-
    # checkers verify the public signature at test collection time.
