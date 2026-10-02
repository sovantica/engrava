#!/usr/bin/env bash
# scripts/run_schema_release_gate.sh — pre-tag schema-compatibility check.
#
# Invoked as a `prepareCmd` of the `@semantic-release/exec` plugin, listed in
# .releaserc.json with the version semantic-release computed
# (`${nextRelease.version}`) as the sole argument. Like every other
# `prepareCmd` in that chain, this runs inside the `prepare` lifecycle step —
# before @semantic-release/git commits, tags, or pushes anything, and before
# @semantic-release/github publishes a GitHub Release. A non-zero exit here
# aborts the whole `prepare` step (semantic-release's own plugin contract:
# see .github/workflows/release.yml's header comment), so a
# schema-incompatible patch release is refused before either the tag or the
# Release exist, not after.
#
# HEAD at this point in the lifecycle is still the ordinary branch tip
# semantic-release started from -- no release commit, no tag, exists yet.
# The last released tag is therefore whatever `git describe` finds reachable
# directly from HEAD, not from HEAD^.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$#" -ne 1 ]; then
  echo "usage: run_schema_release_gate.sh <next-version>" >&2
  exit 2
fi
NEXT_VERSION="$1"

if old_tag=$(git describe --tags --abbrev=0 HEAD 2>/dev/null); then
  python scripts/check_schema_release_gate.py \
    --old-tag "$old_tag" \
    --new-version "$NEXT_VERSION"
else
  echo "No prior release tag reachable from HEAD -- nothing to compare against, skipping the schema-version release gate."
fi
