// Commitlint config — enforces Conventional Commits 1.0.0 + the project scope allowlist.
// Reference: https://commitlint.js.org/
// Used by .github/workflows/commitlint.yml to validate PR titles + commits.
// The scope vocabulary is NOT written here. It lives in commit-scopes.json, the
// single machine-readable source, so that no scope name is ever spelled twice in
// two grammars. Do not add scope names to this file. Two checks sit over it, and
// they prove different things:
//
//   - tests/test_commit_scope_vocabulary.py compares commit-scopes.json against the
//     prose in docs/scopes.md in both directions, and checks that the two lines
//     below are present in this file. Presence only: a tripwire for someone
//     replacing the wiring with a literal list. It cannot show that no second,
//     contradicting line exists — that needs a parser, not a text search.
//   - the "Verify commitlint admits the same scope set as commit-scopes.json" step
//     in .github/workflows/commitlint.yml asks commitlint for the effective rule and
//     fails the job unless it admits the same set of scopes as the two tiers of
//     commit-scopes.json. That is where absence is covered, within the boundary the
//     step's own comment states.
//
// Note also that `ignores` below skips whole messages before any rule runs: any
// message containing the substring "Signed-off-by: dependabot[bot]" anywhere in it,
// from any author and unauthenticated, and any message beginning with merge/revert
// plus whitespace or a colon. For those the scope-enum is never consulted at all.
//
// This is JS (not YAML) because `ignores` requires a predicate function: merge
// commits are not Conventional Commits, and the 4-tier branch flow
// (feature -> release -> dev -> main) produces merge commits at release -> dev
// and dev -> main. commitlint's built-in defaultIgnores only skips standard
// "Merge branch X into Y" / "Merge pull request" messages; the predicate below
// widens that to any subject *starting* with merge or revert. It is a text match on
// the start of the message, so it covers the phrasings this flow actually produces
// and nothing else — a merge message worded some other way is linted normally.

const SCOPES = require("./commit-scopes.json");

module.exports = {
  extends: ["@commitlint/config-conventional"],

  // Skip non-Conventional, git-generated commits. Belt-and-suspenders over
  // commitlint's defaultIgnores (which stays enabled).
  ignores: [
    // Merge/revert commits are not Conventional Commits. This skips any message
    // that *starts* with either word, in any capitalisation, which is what the
    // 4-tier flow produces: "Merge branch…" auto-messages and hand-typed
    // "merge: …" subjects. A message that mentions either word later is linted.
    (message) => /^(merge|revert)[\s:]/i.test(message),
    // Dependabot's auto-generated commit bodies (changelog URLs + the
    // updated-dependencies block) routinely exceed body-max-line-length, so
    // messages carrying its sign-off are skipped and dependency PRs stay green.
    // Note what this does and does not say: it is a substring match anywhere in the
    // message, with no authentication of the author, so it exempts whatever carries
    // that text — not "bot commits".
    (message) => /Signed-off-by: dependabot\[bot\]/.test(message),
  ],

  rules: {
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "perf",
        "docs",
        "style",
        "refactor",
        "test",
        "build",
        "ci",
        "chore",
        "revert",
      ],
    ],

    "scope-enum": [2, "always", [...SCOPES.canonical, ...SCOPES.deprecated]],

    // warning only — some commits (e.g. pure `chore:`) may omit scope
    "scope-empty": [1, "never"],

    "header-max-length": [2, "always", 100],

    "subject-full-stop": [2, "never", "."],

    "body-leading-blank": [2, "always"],

    "footer-leading-blank": [2, "always"],

    "type-empty": [2, "never"],

    "subject-empty": [2, "never"],
  },
};
