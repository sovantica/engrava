#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# The baseline must be a version that is actually on PyPI, otherwise the run
# dies at `pip install`. Bump this with every release, alongside the same pin in
# the ci.yml upgrade-matrix job.
: "${ENGRAVA_UPGRADE_FROM_SPEC:=engrava==0.6.0}"
: "${ENGRAVA_UPGRADE_FROM_EDITABLE:=0}"
: "${ENGRAVA_UPGRADE_TO_EDITABLE:=0}"

cd "$ROOT_DIR"

# Default the upgrade target to a wheel built from this tree, not an
# editable checkout -- an editable install imports straight from the source
# tree, so a packaging-only defect (a missing data file, a broken entry
# point) would be invisible to it, matching the ci.yml upgrade-matrix job.
# That wheel still carries this tree's own pyproject.toml version, not
# whatever version the release pipeline would later bump it to and publish.
# Set ENGRAVA_UPGRADE_TO_SPEC explicitly (and ENGRAVA_UPGRADE_TO_EDITABLE=1)
# for a faster, editable local iteration loop that intentionally skips that
# coverage.
if [ -z "${ENGRAVA_UPGRADE_TO_SPEC:-}" ]; then
    rm -rf "$ROOT_DIR/dist"
    python -m build --wheel "$ROOT_DIR" >/dev/null
    ENGRAVA_UPGRADE_TO_SPEC="$(ls "$ROOT_DIR"/dist/engrava-*.whl | head -n1)"
fi

ENGRAVA_RUN_UPGRADE_MATRIX=1 \
ENGRAVA_UPGRADE_FROM_SPEC="$ENGRAVA_UPGRADE_FROM_SPEC" \
ENGRAVA_UPGRADE_TO_SPEC="$ENGRAVA_UPGRADE_TO_SPEC" \
ENGRAVA_UPGRADE_FROM_EDITABLE="$ENGRAVA_UPGRADE_FROM_EDITABLE" \
ENGRAVA_UPGRADE_TO_EDITABLE="$ENGRAVA_UPGRADE_TO_EDITABLE" \
python -m pytest tests/upgrade/test_upgrade_matrix.py -v