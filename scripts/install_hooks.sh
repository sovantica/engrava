#!/usr/bin/env bash
# scripts/install_hooks.sh — wire this repository's versioned git hooks.
#
# Run via `make install` (or directly: bash scripts/install_hooks.sh) FROM
# THE PRIMARY CHECKOUT ONLY -- see the refusal below for why.
#
# Two things happen here:
#
#   1. core.hooksPath is pointed at .githooks/, using an ABSOLUTE path rather
#      than the literal string ".githooks". A relative value is resolved per
#      checkout: a linked worktree on a branch that lays .githooks/ out
#      differently -- or lacks it entirely -- would silently run that
#      branch's own copy, or nothing at all, with the commit still
#      succeeding. An absolute path resolved once here is the same file from
#      every worktree of this repository. See CONTRIBUTING.md.
#
#   2. commitlint's own dependencies are installed into this repository's
#      node_modules/ (git-ignored), pinned to the same versions
#      .github/workflows/commitlint.yml uses. The hook resolves this
#      installation from every worktree via NODE_PATH -- see
#      .githooks/commit-msg -- so it only needs to exist here, at the
#      primary checkout, and not be reinstalled per worktree. Only the
#      config commitlint enforces (.commitlintrc.js, commit-scopes.json)
#      stays branch-local, by design: it is checked-out, tracked content.
set -euo pipefail

# core.hooksPath lives in this repository's SHARED .git/config -- one value
# for every worktree. Setting it from a linked worktree would repoint every
# checkout's hooks at *this* worktree's .githooks/, a directory that
# disappears the moment this worktree is removed -- silently leaving every
# checkout, including the primary one, with no hook running at all.
# Confirmed live during development: one run from a linked worktree changed
# the value in the real repository's shared config.
#
# Detected via --git-dir vs --git-common-dir rather than by comparing
# working-tree paths: they are equal only in the primary checkout. A linked
# worktree's own --git-dir is a *different*, per-worktree admin directory
# under the shared --git-common-dir.
GIT_DIR="$(git rev-parse --path-format=absolute --git-dir)"
GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY_ROOT="$(dirname "$GIT_COMMON_DIR")"
# Assumes the ordinary layout where --git-common-dir is <primary_root>/.git
# (true for a normal clone/worktree set). --separate-git-dir, a submodule,
# or a bare repository break that assumption; none of those are how
# engrava is used, so this is stated rather than engineered around.

if [ "$GIT_DIR" != "$GIT_COMMON_DIR" ]; then
  cat >&2 <<EOF
install_hooks: refusing to run from a linked worktree.
core.hooksPath is shared by every worktree of this repository. Installing
from here would repoint all of them at this worktree's .githooks/, which
disappears when this worktree does.
Run 'make install' from the primary checkout instead: $PRIMARY_ROOT
EOF
  exit 1
fi

TOPLEVEL="$PRIMARY_ROOT"
HOOKS_DIR="$TOPLEVEL/.githooks"

if [ ! -d "$HOOKS_DIR" ]; then
  echo "install_hooks: $HOOKS_DIR not found" >&2
  exit 1
fi

chmod +x "$HOOKS_DIR"/* 2>/dev/null || true
git config core.hooksPath "$HOOKS_DIR"
echo "Git hooks wired: core.hooksPath = $HOOKS_DIR"

if command -v npm >/dev/null 2>&1; then
  echo ">> Installing commitlint (used by the commit-msg hook)"
  ( cd "$TOPLEVEL" && npm install --no-save @commitlint/cli@19 @commitlint/config-conventional@19 )
else
  cat >&2 <<'EOF'
install_hooks: npm not found -- commitlint will not be installed.
The commit-msg hook fails CLOSED (refuses commits) until Node/npm is
available and this script is re-run. See CONTRIBUTING.md.
EOF
fi
