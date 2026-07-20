"""Regenerate (or verify) the checked-in search-semantics goldens.

The two goldens under ``tests/search_contract/goldens/`` pin retrieval
*semantics*: a byte-identical FTS5 expert-normalizer parity map and a frozen
hybrid ranked-result list. The tests only read them — they never rewrite a
golden on mismatch, because a self-rewriting golden is coverage-padding, not a
check. This is the explicit, reviewed path a maintainer runs after an
*intended* retrieval change:

    python scripts/regenerate_search_goldens.py            # rewrite the goldens
    python scripts/regenerate_search_goldens.py --check    # fail if stale (CI)

Regenerating changes tracked files, so the diff is reviewed like any other code
change before it lands.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.search_contract.golden_fixtures import (
    EXPERT_NORMALIZATION_GOLDEN_PATH,
    HYBRID_RANKED_GOLDEN_PATH,
    render_expert_normalization_golden,
    render_hybrid_ranked_golden,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the goldens are up to date and exit non-zero if stale, without writing.",
    )
    return parser.parse_args()


def _render_all() -> list[tuple[Path, str]]:
    """Render every golden to its canonical on-disk text."""
    return [
        (EXPERT_NORMALIZATION_GOLDEN_PATH, render_expert_normalization_golden()),
        (HYBRID_RANKED_GOLDEN_PATH, asyncio.run(render_hybrid_ranked_golden())),
    ]


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    targets = _render_all()

    if args.check:
        stale = [
            path
            for path, content in targets
            if not path.exists() or path.read_text(encoding="utf-8") != content
        ]
        for path in stale:
            print(f"STALE: {path.relative_to(REPOSITORY_ROOT)}")
        if stale:
            print("Search goldens are out of date; run: python scripts/regenerate_search_goldens.py")
            return 1
        print("Search goldens are up to date.")
        return 0

    for path, content in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
