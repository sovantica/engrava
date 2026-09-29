#!/usr/bin/env bash
# scripts/check_hooks_active.sh — is core.hooksPath actually wired to THIS
# repository's own hooks, and does the commit-msg hook's grammar check
# actually resolve from here?
#
# core.hooksPath has gone silently unset in this repository family before,
# and a hook that has quietly stopped running is worse than no hook: the
# absence of failures reads as a pass. This runs from `make gate` so a
# contributor sees it before that can happen unnoticed, rather than
# discovering it the day a bad commit message goes through.
#
# Wiring is not enough, and neither is "some hook is wired". A hook that is
# wired, executable, and cannot resolve commitlint used to warn and let the
# commit through -- a gate reporting its health by existing, exactly what
# this gate exists to catch. And a value that merely points at SOME
# absolute directory containing an executable named commit-msg would pass
# even if that hook belongs to something else entirely and never checks a
# thing. So this verifies four separate claims: the value is this
# repository's OWN .githooks at the primary checkout, the hook there is
# executable, commitlint actually resolves its configuration from this
# checkout's working directory, the same way the real hook does, and the
# hook itself, when run, accepts a well-formed message and rejects a
# malformed one. A hook file replaced by `exit 0` is wired and executable,
# and checks nothing; the last claim is the one it fails.
set -euo pipefail

HOOKS_PATH="$(git config --get core.hooksPath || true)"

if [ -z "$HOOKS_PATH" ]; then
  echo "✗ core.hooksPath is not set -- git hooks are NOT active. Run 'make install'." >&2
  exit 1
fi

case "$HOOKS_PATH" in
  /*) : ;;
  *)
    echo "✗ core.hooksPath ('$HOOKS_PATH') is relative -- it resolves per checkout" >&2
    echo "  and can silently point at nothing from another worktree or branch." >&2
    echo "  Re-run 'make install', which sets an absolute path." >&2
    exit 1
    ;;
esac

# The primary checkout's root, from the one shared git-dir every worktree of
# this repository points at (its parent, regardless of which worktree this
# script itself runs from -- see .githooks/commit-msg for the same
# resolution). Split into two commands rather than
# `dirname "$(git rev-parse ...)"` in one line: dirname happily turns a
# failed substitution's empty output into ".", which would let this check
# pass (against "./node_modules/.bin/commitlint", if that happened to
# exist) instead of failing when the rev-parse it depends on fails.
GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY_ROOT="$(dirname "$GIT_COMMON_DIR")"
# Assumes the ordinary layout where --git-common-dir is <primary_root>/.git
# (true for a normal clone/worktree set). --separate-git-dir, a submodule,
# or a bare repository break that assumption; none of those are how
# engrava is used, so this is stated rather than engineered around.
EXPECTED_HOOKS_PATH="$PRIMARY_ROOT/.githooks"

if [ "$HOOKS_PATH" != "$EXPECTED_HOOKS_PATH" ]; then
  echo "✗ core.hooksPath is '$HOOKS_PATH', not this repository's own" >&2
  echo "  $EXPECTED_HOOKS_PATH. An unrelated hook at an unrelated path would" >&2
  echo "  otherwise pass this check while commits go unchecked." >&2
  echo "  Re-run 'make install' from the primary checkout." >&2
  exit 1
fi

if [ ! -x "$HOOKS_PATH/commit-msg" ]; then
  echo "✗ core.hooksPath is set to '$HOOKS_PATH' but commit-msg there is missing" >&2
  echo "  or not executable. Re-run 'make install'." >&2
  exit 1
fi

# The hook resolves commitlint from the PRIMARY checkout's node_modules --
# already established above to be the same directory core.hooksPath points
# at, since the check above requires HOOKS_PATH to equal the primary's own
# .githooks.
COMMITLINT_BIN="$PRIMARY_ROOT/node_modules/.bin/commitlint"

if [ ! -x "$COMMITLINT_BIN" ]; then
  echo "✗ commitlint is not installed at $PRIMARY_ROOT/node_modules." >&2
  echo "  The commit-msg hook will refuse every commit until 'make install'" >&2
  echo "  runs from the primary checkout: $PRIMARY_ROOT" >&2
  exit 1
fi

# Actually resolve config from THIS checkout's working directory -- the same
# thing the hook does for every commit here. --print-config exercises the
# same "extends" resolution that fails silently-from-nowhere when the
# linter's own dependencies (e.g. @commitlint/config-conventional) are not
# reachable from this directory, which is exactly the failure mode that let
# a bad commit through with a green activity check.
if ! CONFIG_OUTPUT="$(NODE_PATH="$PRIMARY_ROOT/node_modules" "$COMMITLINT_BIN" --print-config json 2>&1)"; then
  echo "✗ commitlint does not resolve its configuration from this checkout:" >&2
  echo "$CONFIG_OUTPUT" | sed 's/^/  /' >&2
  exit 1
fi

# Run the hook itself from the work tree root with the path of a message file
# as its only argument. Two message files in a private scratch directory, one
# the repository's own grammar accepts and one it rejects. The hook chains to
# an executable at <git-common-dir>/hooks/commit-msg, if there is one, and a
# nonzero exit from that guard makes the hook reject the message.
#
# The hook's own grammar check is skipped on some inputs, and the inputs below
# are chosen to reach it: neither subject is one that .commitlintrc.js ignores,
# and no true merge may be in progress (the hook skips its own grammar check
# when MERGE_HEAD exists). The scope "docs" is in
# commit-scopes.json, so the good message also produces no scope warning;
# "wibble" is outside the type-enum.
MERGE_HEAD="$(git rev-parse --git-path MERGE_HEAD)"
if [ -f "$MERGE_HEAD" ]; then
  echo "✗ a merge is in progress ($MERGE_HEAD exists)." >&2
  echo "  The commit-msg hook skips its own grammar check for a true merge commit," >&2
  echo "  so it cannot be exercised now. Conclude or abort the merge and re-run." >&2
  exit 1
fi

WORK_TREE="$(git rev-parse --show-toplevel)"
HOOK="$HOOKS_PATH/commit-msg"
GOOD_MESSAGE="docs(docs): describe the thing"
BAD_MESSAGE="wibble(docs): describe the thing"

if ! SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/hooks-active.XXXXXX")"; then
  echo "✗ could not create a scratch directory to run the commit-msg hook in." >&2
  exit 1
fi
trap 'rm -rf "$SCRATCH"' EXIT
printf '%s\n' "$GOOD_MESSAGE" > "$SCRATCH/good.txt"
printf '%s\n' "$BAD_MESSAGE" > "$SCRATCH/bad.txt"

if ! GOOD_OUTPUT="$(cd "$WORK_TREE" && "$HOOK" "$SCRATCH/good.txt" 2>&1)"; then
  echo "✗ the commit-msg hook at $HOOK rejected a well-formed message:" >&2
  echo "  $GOOD_MESSAGE" >&2
  if [ -n "$GOOD_OUTPUT" ]; then echo "$GOOD_OUTPUT" | sed 's/^/  /' >&2; fi
  exit 1
fi

if BAD_OUTPUT="$(cd "$WORK_TREE" && "$HOOK" "$SCRATCH/bad.txt" 2>&1)"; then
  echo "✗ the commit-msg hook at $HOOK accepted a message whose type is not allowed:" >&2
  echo "  $BAD_MESSAGE" >&2
  if [ -n "$BAD_OUTPUT" ]; then echo "$BAD_OUTPUT" | sed 's/^/  /' >&2; fi
  exit 1
fi

echo "✓ Git hooks active: core.hooksPath = $HOOKS_PATH (this repository's own)"
echo "✓ commitlint resolves from this checkout via $COMMITLINT_BIN"
echo "✓ the commit-msg hook accepts a well-formed message and rejects a malformed one"
