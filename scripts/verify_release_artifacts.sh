#!/usr/bin/env bash
# scripts/verify_release_artifacts.sh — the release gate that must run, and
# must pass, before a tag or a GitHub Release can come into existence.
#
# Invoked as a `prepareCmd` of the `@semantic-release/exec` plugin, listed in
# .releaserc.json *after* the plugin entry that bumps pyproject.toml's version
# and *before* @semantic-release/git, with the version semantic-release
# computed (`${nextRelease.version}`) as the sole argument. semantic-release
# runs a lifecycle step's plugins in array order and aborts the whole step on
# the first failure (it does not run the remaining plugins for that step) —
# so a non-zero exit here stops @semantic-release/git from ever committing or
# pushing, which in turn means semantic-release's own tag-and-push (which it
# always performs immediately after the `prepare` step, before any `publish`
# plugin such as @semantic-release/github runs) never happens either. No
# separate flag or job condition is needed to enforce that ordering; it falls
# out of the plugin contract itself.
#
# By the time this script runs, pyproject.toml already carries the version
# that is about to be tagged — scripts/bump_pyproject_version.py wrote it to
# disk (and already refused to continue if that write didn't happen), and
# nothing has committed it yet. scripts/verify_wheel_data.py performs the
# release's one and only `python -m build`, from that exact on-disk state.
#
# What runs after the build establishes, in order, that there is exactly one
# thing to verify and ship, that it is the right thing, and that it is safe
# to ship: scripts/verify_dist_cardinality.py asserts dist/ holds exactly one
# wheel and one sdist and nothing else — twine check, the sha256 manifest,
# the artifact upload, and pypa's publish action all operate on the whole
# directory, not on whichever pair a later glob happens to pick, so that
# pair has to be proven unique before anything trusts it. Only then does
# scripts/verify_artifact_version.py re-derive the version from the built
# wheel/sdist metadata itself (not their filenames, which would agree with a
# stale build too) and compare it against the version being tagged, so a
# build that silently drifted from what was meant to ship is caught here
# rather than shipped.
#
# dist/SHA256SUMS, written last, is the byte-identity record the publish job
# checks before it uploads: the publish job never rebuilds, it only verifies
# that what it is about to ship still hashes to what this script verified.
#
# The very first thing this script does, ahead of even the smoke gate, is
# scripts/verify_release_content_scan.sh — the one call site both this gate
# and this repository's ordinary secret scan use for text the ordinary scan
# never reads at all: the squash-merge commit message this release was
# built from, and CHANGELOG.md as @semantic-release/changelog already wrote
# it to disk (which is, by construction, the same text
# @semantic-release/git embeds in the release commit message and
# @semantic-release/github publishes as the Release body — see that script
# for why scanning the file covers all three). It runs first because it is
# the cheapest check here and needs none of the build below to have
# happened.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$#" -ne 1 ]; then
  echo "usage: verify_release_artifacts.sh <expected-version>" >&2
  exit 2
fi
EXPECTED_VERSION="$1"

echo "== Scanning for secrets the ordinary scan never reads (commit messages, generated changelog) =="
bash scripts/verify_release_content_scan.sh

echo "== Pre-publish smoke gate =="
python scripts/check_smoke_gate.py

echo "== Build (once) + verify wheel/sdist package data =="
python scripts/verify_wheel_data.py

echo "== Verify dist/ holds exactly one wheel and one sdist, nothing else =="
python scripts/verify_dist_cardinality.py

echo "== Verify the built artefacts embed the version being tagged =="
python scripts/verify_artifact_version.py "$EXPECTED_VERSION"

echo "== twine check on the artefacts that will ship =="
python -m twine check dist/*

echo "== Recording artefact hashes for the publish job's byte-identity check =="
# A manifest left over from an earlier attempt would sit inside the same
# glob the next line hashes: bash expands `*` before the `>SHA256SUMS`
# redirect opens the file, so a stale SHA256SUMS gets included, then
# truncated by that same redirect — recording the hash of an empty file
# against its own name. dist/ is fresh here only because
# verify_wheel_data.py wipes and rebuilds it above; remove the manifest
# explicitly anyway so this script does not depend on that being true.
rm -f dist/SHA256SUMS
(
  cd dist
  sha256sum -- * >SHA256SUMS
  cat SHA256SUMS
)
