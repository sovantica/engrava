// Commitlint config — enforces Conventional Commits 1.0.0 + the project scope allowlist.
// Reference: https://commitlint.js.org/
// Used by .github/workflows/commitlint.yml to validate PR titles + commits.
// Scope list synced with docs/scopes.md (human-readable version).
//
// This is JS (not YAML) because `ignores` requires a predicate function: merge
// commits are not Conventional Commits, and the 4-tier branch flow
// (feature -> release -> dev -> main) produces merge commits at release -> dev
// and dev -> main. commitlint's built-in defaultIgnores only skips standard
// "Merge branch X into Y" / "Merge pull request" messages; an explicit ignore
// for any "Merge "-prefixed (and "Revert ") subject keeps the lint green for
// every merge commit regardless of how the message is phrased.

module.exports = {
  extends: ["@commitlint/config-conventional"],

  // Skip non-Conventional, git-generated commits. Belt-and-suspenders over
  // commitlint's defaultIgnores (which stays enabled).
  ignores: [
    (message) =>
      message.startsWith("Merge ") || message.startsWith("Revert "),
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

    "scope-enum": [
      2,
      "always",
      [
        "core",
        "domain",
        "infra",
        "dreaming",
        "search",
        "embedding",
        "mindql",
        "storage",
        "extensions",
        "cli",
        "config",
        "docs",
        "bench",
        "deps",
        "release",
        "ci",
        "build",
      ],
    ],

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
