#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# The baseline must be a version that is actually on PyPI, otherwise the run
# dies at `pip install`. Bump this with every release, alongside the same pin in
# the ci.yml upgrade-matrix job.
: "${ENGRAVA_UPGRADE_FROM_SPEC:=engrava==0.5.0}"
: "${ENGRAVA_UPGRADE_TO_SPEC:=.}"
: "${ENGRAVA_UPGRADE_FROM_EDITABLE:=0}"
: "${ENGRAVA_UPGRADE_TO_EDITABLE:=1}"

cd "$ROOT_DIR"

ENGRAVA_RUN_UPGRADE_MATRIX=1 \
ENGRAVA_UPGRADE_FROM_SPEC="$ENGRAVA_UPGRADE_FROM_SPEC" \
ENGRAVA_UPGRADE_TO_SPEC="$ENGRAVA_UPGRADE_TO_SPEC" \
ENGRAVA_UPGRADE_FROM_EDITABLE="$ENGRAVA_UPGRADE_FROM_EDITABLE" \
ENGRAVA_UPGRADE_TO_EDITABLE="$ENGRAVA_UPGRADE_TO_EDITABLE" \
python -m pytest tests/upgrade/test_upgrade_matrix.py -v