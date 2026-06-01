"""``python -m engrava.benchmarks.longmemeval`` entry point."""

from __future__ import annotations

import sys

from engrava.benchmarks.longmemeval.cli import main

if __name__ == "__main__":
    sys.exit(main())
