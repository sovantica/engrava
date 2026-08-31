#!/usr/bin/env bash
# scripts/verify_release_content_scan.sh — scans the two things the merge-time
# secret scan never reads, before either can reach a published surface.
#
# `.github/workflows/secret-scan.yml`'s Gitleaks scan is a required check on
# the `dev` ruleset, so the *files* landing on `dev` have been scanned before
# they get there. Two things are still unscanned when this repository's
# release pipeline starts building from that same push:
#
#   1. The message of the commit a squash merge creates. That commit is
#      authored in the merge dialog, after the PR's checks (including the
#      Gitleaks scan) already ran, so nothing has ever read its text. Measured
#      directly (see the commit that introduced this script): gitleaks' `git`
#      mode diffs each commit's *changed file content* — it does not inspect
#      commit messages at all, on any commit, scanned or not. So this is not
#      a gap that closing races or widening triggers fixes; the tool itself
#      never looks at this text, in any mode this repository already runs.
#
#   2. `CHANGELOG.md`, written to disk by `@semantic-release/changelog`
#      before this script runs (see scripts/verify_release_artifacts.sh,
#      which invokes this script first, ahead of the build). Its released-
#      version section is generated from the same commit messages, including
#      the one squash-merge message this repository never scans, so a
#      secret-shaped value there reaches this file even when the exact same
#      value in a diff would have been caught. This file ships inside the
#      sdist (MANIFEST.in), and its content is, by construction, the same
#      release notes `@semantic-release/git` embeds verbatim in the release
#      commit message (`.releaserc.json`'s `message` template) and
#      `@semantic-release/github` publishes verbatim as the GitHub Release
#      body — one generated text, three destinations. Gating this one file
#      before it is committed, tagged or announced therefore gates all three;
#      there is no destination-specific scan to add for the other two, since
#      neither exists as its own artifact before `@semantic-release/git`
#      commits — checked directly in the commit that added this script,
#      against a synthetic-secret CHANGELOG.md entry.
#
# Uses the exact gitleaks binary scripts/install_gitleaks.sh pins and
# verifies — the release workflow installs it into the repository root
# before `npx semantic-release` runs (see .github/workflows/release.yml),
# the same way secret-scan.yml does for its own job. Two independently
# pinned gitleaks binaries in this repository would be a second thing that
# could silently drift from the first; there is exactly one.
#
# `gitleaks detect --no-git` (not `gitleaks git`) is used throughout: it
# scans the literal bytes given to it, with no notion of commits or diffs,
# which is what both inputs below need — a file already on disk and a
# synthetic text file this script assembles from `git log` output are
# neither a git repository nor a diff.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GITLEAKS_BIN="./gitleaks"
if [ ! -x "$GITLEAKS_BIN" ]; then
  echo "verify_release_content_scan: $GITLEAKS_BIN not found or not executable." >&2
  echo "  The release workflow must install it (scripts/install_gitleaks.sh)" >&2
  echo "  before running semantic-release. Refusing rather than skipping the scan." >&2
  exit 1
fi

status=0

# --- 1. The commit messages this release is actually made of -------------
#
# At this point in the `prepare` lifecycle, semantic-release has not yet
# created this release's tag (that happens only after every prepareCmd
# below this one succeeds), so "the previous release" is exactly the
# nearest reachable tag from HEAD. The very first release, before any tag
# exists, falls back to the repository's whole reachable history — a
# one-time, strictly larger scan rather than an unguarded gap.
PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
if [ -n "$PREV_TAG" ]; then
  COMMIT_RANGE="${PREV_TAG}..HEAD"
else
  COMMIT_RANGE="HEAD"
fi

MESSAGES_FILE="$(mktemp)"
trap 'rm -f "$MESSAGES_FILE"' EXIT
# %x00 between messages: a message containing a line that looks like another
# commit's header can't be misread as a boundary the way a blank-line
# separator could.
git log --format='%B%x00' "$COMMIT_RANGE" > "$MESSAGES_FILE"

echo "== Scanning commit messages for $COMMIT_RANGE (includes any squash-merge message) =="
if ! "$GITLEAKS_BIN" detect --no-git --source "$MESSAGES_FILE" --redact --verbose --exit-code 1; then
  echo "verify_release_content_scan: a commit message in $COMMIT_RANGE looks like a secret." >&2
  status=1
fi

# --- 2. CHANGELOG.md as written to disk for this release ------------------
if [ ! -f CHANGELOG.md ]; then
  echo "verify_release_content_scan: CHANGELOG.md not found — expected" >&2
  echo "  @semantic-release/changelog to have written it before this script runs." >&2
  exit 1
fi

echo "== Scanning CHANGELOG.md (covers the Release body and the release commit message, which embed the same generated notes) =="
if ! "$GITLEAKS_BIN" detect --no-git --source CHANGELOG.md --redact --verbose --exit-code 1; then
  echo "verify_release_content_scan: CHANGELOG.md looks like it contains a secret." >&2
  status=1
fi

exit "$status"
