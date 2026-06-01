"""``python -m engrava.benchmarks.synthetic`` entry point."""

from __future__ import annotations

import sys

from engrava.benchmarks.synthetic.runner import main

if __name__ == "__main__":
    sys.exit(main())
