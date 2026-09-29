#!/usr/bin/env bash
# scripts/install_gitleaks.sh — the one place that names which gitleaks
# binary this repository trusts and how that trust is checked.
#
# Two jobs need the exact same gitleaks: .github/workflows/secret-scan.yml
# (scans push/PR history) and .github/workflows/release.yml, whose
# scripts/verify_release_content_scan.sh scans the release commit range and
# the generated changelog before a release is tagged. Both workflows install
# it by running this script rather than repeating the pin: a second inline
# copy of the version and its sha256 would be a second place that could
# drift from the first — quietly trusting a different binary in the one job
# that gates publication.
#
# Neither gitleaks-action nor any other third-party Action is used to
# install it: this repository's Actions policy allows only GitHub-owned
# (actions/*) and verified-publisher actions, and gitleaks' own action is
# neither. Downloading the official release binary directly and checking it
# against a checksum pinned here — rather than one gitleaks' own release
# page serves at fetch time — gives a reproducible, offline-verifiable
# install without taking a dependency on a non-allowed Action.
#
# Usage: scripts/install_gitleaks.sh [dest-dir]
#   dest-dir defaults to the current directory. The binary is written there
#   as "gitleaks" (Linux x86_64 only — every current call site runs on
#   ubuntu-latest; a call site on another OS/arch would need this script
#   extended, not a second copy of it written elsewhere).
set -euo pipefail

GITLEAKS_VERSION="8.30.1"
GITLEAKS_SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"

DEST_DIR="${1:-.}"
mkdir -p "$DEST_DIR"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

ARCHIVE="$WORK_DIR/gitleaks.tar.gz"
URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"

echo "install_gitleaks: downloading gitleaks v${GITLEAKS_VERSION}" >&2
curl -sSfL "$URL" -o "$ARCHIVE"
echo "${GITLEAKS_SHA256}  ${ARCHIVE}" | sha256sum -c -

tar -xzf "$ARCHIVE" -C "$WORK_DIR" gitleaks
chmod +x "$WORK_DIR/gitleaks"
mv "$WORK_DIR/gitleaks" "$DEST_DIR/gitleaks"

"$DEST_DIR/gitleaks" version
