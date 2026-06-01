"""Frozen-dataset guards for ``synthetic-v1.json``.

Two binding invariants the CI MUST enforce on every run:

* ``test_frozen_synthetic_v1_unchanged`` — the committed JSON file
  is byte-identical to live regeneration from the recorded seed and
  parameters.  Catches accidental edits to the data file (which
  silently change the corpus the recall numbers refer to) and
  surfaces generator-algorithm changes as an explicit test failure
  forcing a conscious decision: regenerate (commit new bytes) or
  revert the generator change.
* ``test_synthetic_v1_is_bundled`` — ``importlib.resources`` can
  reach the file from the installed package.  Pins the additive
  ``[tool.setuptools.package-data]`` line that makes the wheel ship
  the JSON together with the code.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

from engrava.benchmarks.synthetic.generate import (
    dataset_to_json,
    generate_dataset,
)

# Generation parameters used to freeze ``synthetic-v1``.  Keep these
# documented inline; the runner CLI reads from the same constants
# (``_DEFAULT_SEED`` / ``_DEFAULT_N_CONVERSATIONS`` / ``_DEFAULT_TURNS``
# / ``_DEFAULT_DENSITY`` in ``runner.py``) so the freeze guard cannot
# drift from the binding default invocation.
_FROZEN_SEED = 20260508
_FROZEN_N_CONVERSATIONS = 50
_FROZEN_AVG_TURNS = 70
_FROZEN_DISTRACTION_DENSITY = 0.4

_FROZEN_PACKAGE = "engrava.benchmarks.synthetic.datasets"
_FROZEN_FILENAME = "synthetic-v1.json"

_REPO_FROZEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engrava"
    / "benchmarks"
    / "synthetic"
    / "datasets"
    / _FROZEN_FILENAME
)


class TestFrozenDatasetByteIdentity:
    """The committed JSON file matches the generator's live output."""

    def test_frozen_synthetic_v1_unchanged(self) -> None:
        # Read the committed file via the repo path (the wheel-side
        # bundling test below covers the installed location), compare
        # byte-for-byte against the live regeneration from the same
        # seed and parameters.
        expected = _REPO_FROZEN_PATH.read_text(encoding="utf-8")
        regenerated = dataset_to_json(
            generate_dataset(
                seed=_FROZEN_SEED,
                n_conversations=_FROZEN_N_CONVERSATIONS,
                avg_turns_per_conversation=_FROZEN_AVG_TURNS,
                distraction_density=_FROZEN_DISTRACTION_DENSITY,
            ),
        )
        if expected != regenerated:
            # Surface the first divergence so a reviewer can tell at
            # a glance whether the drift is cosmetic (whitespace) or
            # structural (different scenario mix / slot values).
            first_diff_idx = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(expected, regenerated, strict=False),
                    )
                    if a != b
                ),
                min(len(expected), len(regenerated)),
            )
            msg = (
                f"synthetic-v1.json drifted from its frozen seed output.\n"
                f"  expected length: {len(expected)}\n"
                f"  generated length: {len(regenerated)}\n"
                f"  first divergence at byte {first_diff_idx}.\n"
                f"  expected context: "
                f"{expected[max(0, first_diff_idx - 60) : first_diff_idx + 60]!r}\n"
                f"  generated context: "
                f"{regenerated[max(0, first_diff_idx - 60) : first_diff_idx + 60]!r}\n"
                "  Either the generator algorithm changed (commit new "
                "bytes via `python -m engrava.benchmarks.synthetic "
                "--regenerate`) OR deterministic serialization broke."
            )
            raise AssertionError(msg)

    def test_frozen_synthetic_v1_is_valid_json(self) -> None:
        payload = json.loads(_REPO_FROZEN_PATH.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert len(payload) == _FROZEN_N_CONVERSATIONS

    def test_frozen_synthetic_v1_size_within_reasonable_bound(self) -> None:
        # Pre-publish sanity: the wheel ships this file; an accidental
        # 100 MB explosion (e.g. someone bumps ``n_conversations``
        # into the thousands) would balloon the package.  Cap at 5
        # MiB so legitimate corpus growth into the low megabytes
        # still fits.
        size_bytes = _REPO_FROZEN_PATH.stat().st_size
        max_bytes = 5 * 1024 * 1024
        assert size_bytes <= max_bytes, (
            f"synthetic-v1.json is {size_bytes / 1024 / 1024:.2f} MiB — "
            f"exceeds the {max_bytes / 1024 / 1024:.0f} MiB pre-publish "
            f"ceiling.  Consider trimming the dataset before committing."
        )


class TestFrozenDatasetBundling:
    """``importlib.resources`` reaches the file from the installed package."""

    def test_synthetic_v1_is_bundled(self) -> None:
        resource = importlib.resources.files(_FROZEN_PACKAGE).joinpath(
            _FROZEN_FILENAME,
        )
        assert resource.is_file(), (
            "synthetic-v1.json not reachable via importlib.resources — "
            "package-data wiring is broken."
        )
        content = resource.read_text(encoding="utf-8")
        assert len(content) > 1000, "synthetic-v1.json is suspiciously small"
        # Quick sanity: the bundled bytes must match the repo bytes.
        # When running from a source checkout these are the same file,
        # but the assertion guards against accidental drift if the
        # ``importlib.resources`` lookup ever resolves to a different
        # copy (e.g. a stale site-packages install on the test
        # machine).
        assert content == _REPO_FROZEN_PATH.read_text(encoding="utf-8")

    def test_bundled_dataset_loads_through_runner_loader(self) -> None:
        # End-to-end: the same loader the CLI uses reads the bundled
        # JSON without error and yields the expected conversation
        # count.
        from engrava.benchmarks.synthetic.runner import _load_frozen_dataset

        loaded = _load_frozen_dataset()
        assert len(loaded) == _FROZEN_N_CONVERSATIONS
