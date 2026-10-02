"""Tests for the main-carries-the-released-tag check.

Most cases drive the pure logic functions directly with crafted values —
no git needed. Two kinds of tests need a real git repository instead of
crafted values or monkeypatched functions: the ``is_ancestor`` /
``assert_ref_is_qualified_and_exists`` behaviour depends on git's own exit
codes and ref-format rules, which a stub cannot reproduce faithfully. Those build
a disposable repository under ``tmp_path`` and point the module at it by
monkeypatching ``REPO_ROOT`` -- the module resolves that constant once at
import time, so this is the seam, and it keeps production code unchanged.

``TestFailabilityOnRealHistory`` is separate again: it runs the check
against *this* repository's own history, because a check like this is only
worth anything once it has actually been seen red as well as green. Those
cases are skipped when the required history is absent (e.g. a shallow CI
checkout), which is why the other two kinds of git-backed tests above exist
as unconditional coverage of the same code paths.
"""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_main_carries_the_released_tag.py"


@pytest.fixture
def gate_module() -> object:
    """Load ``scripts/check_main_carries_the_released_tag.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location(
        "check_main_carries_the_released_tag", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        msg = f"could not load main-carries-the-released-tag module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ref_exists(ref: str) -> bool:
    completed = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "--verify", "--quiet", ref],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _newest_release_tag_in_this_repo() -> tuple[str, tuple[int, int, int]]:
    """Return the highest ``vMAJOR.MINOR.PATCH`` tag here, read without the gate's helpers."""
    listed = subprocess.run(
        # Plumbing, not `git tag --list`: column.tag/column.ui can pack several
        # names onto one line of the porcelain listing. Bytes, decoded per name
        # with replacement, so a tag name that is not UTF-8 cannot raise here.
        ["git", "for-each-ref", "--format=%(refname:lstrip=2)", "refs/tags/v*"],  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\n")
    versions = {}
    for raw in listed:
        tag = raw.decode("utf-8", errors="replace")
        match = re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag)
        if match:
            versions[tag] = (int(match[1]), int(match[2]), int(match[3]))
    newest = max(versions, key=versions.__getitem__)
    return newest, versions[newest]


def _init_disposable_repo(path: Path) -> None:
    """Initialise an empty git repository at ``path`` with a local commit identity."""
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)  # noqa: S603, S607
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],  # noqa: S607
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)  # noqa: S607


def _commit(path: Path, message: str) -> str:
    """Create an empty commit in the repo at ``path`` and return its SHA."""
    subprocess.run(  # noqa: S603
        ["git", "commit", "--quiet", "--allow-empty", "-m", message],  # noqa: S607
        cwd=path,
        check=True,
    )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


REQUIRES_V0_6_0 = pytest.mark.skipif(
    not _ref_exists("refs/tags/v0.6.0"),
    reason="requires the v0.6.0 tag to be present in this checkout",
)
REQUIRES_ORIGIN_MAIN = pytest.mark.skipif(
    not _ref_exists("refs/remotes/origin/main"),
    reason="requires an 'origin/main' remote-tracking ref in this checkout",
)
# The exact first parent of the merge commit that brought v0.6.0 into main
# (github.com/sovantica/engrava commit 8c044e2, "Merge pull request #49 from
# sovantica/chore/forward-merge-the-0-6-0-release"). This is the state main
# was in before the forward merge landed -- not a synthetic scenario.
PRE_FORWARD_MERGE_COMMIT = "4185a5d8a33dbd591125aa4ccd476c2455652b3d"
REQUIRES_PRE_FORWARD_MERGE_COMMIT = pytest.mark.skipif(
    not _ref_exists(PRE_FORWARD_MERGE_COMMIT),
    reason="requires the pre-forward-merge commit to be present in this checkout",
)


class TestParseTagVersion:
    def test_parses_a_well_formed_tag(self, gate_module: object) -> None:
        assert gate_module.parse_tag_version("v1.2.3") == (1, 2, 3)  # type: ignore[attr-defined]

    def test_rejects_a_tag_without_the_v_prefix(self, gate_module: object) -> None:
        assert gate_module.parse_tag_version("1.2.3") is None  # type: ignore[attr-defined]

    def test_rejects_a_prerelease_suffix(self, gate_module: object) -> None:
        assert gate_module.parse_tag_version("v1.2.3-rc.1") is None  # type: ignore[attr-defined]

    def test_rejects_an_unrelated_tag(self, gate_module: object) -> None:
        assert gate_module.parse_tag_version("nightly") is None  # type: ignore[attr-defined]

    def test_rejects_a_leading_zero_component(self, gate_module: object) -> None:
        # v01.0.0 is not valid semantic versioning -- a leading zero must
        # not be silently accepted as a spelling of "1".
        assert gate_module.parse_tag_version("v01.0.0") is None  # type: ignore[attr-defined]

    def test_accepts_a_legitimate_zero_component(self, gate_module: object) -> None:
        # A bare "0" (no leading zero *on* it) is a legitimate component,
        # e.g. a pre-1.0 release -- this must still parse.
        assert gate_module.parse_tag_version("v0.1.0") == (0, 1, 0)  # type: ignore[attr-defined]


class TestNewestVersionTag:
    def test_picks_the_highest_semver(self, gate_module: object) -> None:
        tags = ["v0.2.0", "v0.10.0", "v0.9.0"]
        assert gate_module.newest_version_tag(tags) == "v0.10.0"  # type: ignore[attr-defined]

    def test_a_malformed_leading_zero_tag_does_not_shadow_the_real_one(
        self, gate_module: object
    ) -> None:
        # The malformed spelling v01.0.0 must never be a candidate at all,
        # regardless of list order: if it parsed to (1, 0, 0) it would tie
        # with v1.0.0, and max() keeps whichever of two equal keys it saw
        # first.
        assert gate_module.newest_version_tag(["v01.0.0", "v1.0.0"]) == "v1.0.0"  # type: ignore[attr-defined]
        assert gate_module.newest_version_tag(["v1.0.0", "v01.0.0"]) == "v1.0.0"  # type: ignore[attr-defined]

    def test_ignores_non_version_tags(self, gate_module: object) -> None:
        tags = ["v0.5.0", "nightly", "v0.6.0-rc.1"]
        assert gate_module.newest_version_tag(tags) == "v0.5.0"  # type: ignore[attr-defined]

    def test_major_version_takes_precedence_over_a_larger_minor(self, gate_module: object) -> None:
        # A comparison that orders minor before major would wrongly pick
        # v0.10.0 here: 10 > 0 in the minor slot, but 1 > 0 in the major
        # slot must decide first.
        tags = ["v0.10.0", "v1.0.0"]
        assert gate_module.newest_version_tag(tags) == "v1.0.0"  # type: ignore[attr-defined]

    def test_patch_version_decides_when_major_and_minor_are_equal(
        self, gate_module: object
    ) -> None:
        tags = ["v0.7.0", "v0.7.1", "v0.6.9"]
        assert gate_module.newest_version_tag(tags) == "v0.7.1"  # type: ignore[attr-defined]

    def test_raises_when_nothing_matches(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.newest_version_tag(["nightly", "v1.2.3-rc.1"])  # type: ignore[attr-defined]


class TestRunGateWithStubbedReads:
    """Drive run_gate() with monkeypatched git calls.

    Three of the four reads ``run_gate()`` can make against git are stubbed here,
    including ``assert_ref_is_qualified_and_exists`` -- not just
    ``list_git_tags`` and ``is_ancestor``. That third stub is not
    decorative: ``run_gate()`` calls ``assert_ref_is_qualified_and_exists``
    *first*, and it shells out to ``git rev-parse --verify`` against the
    real ``REPO_ROOT`` (this repository's own checkout, since these tests
    never monkeypatch ``REPO_ROOT`` the way the disposable-repository tests
    elsewhere in this file do). Without the stub, "refs/heads/main" here
    would resolve against whatever branches this checkout happens to have,
    so the tests would pass only where a local ``refs/heads/main`` exists.
    A checkout with no local ``refs/heads/main`` (a shallow single-branch
    clone of another branch, for example) would make the tests that reach
    a real assertion below raise ``GateInputError: 'refs/heads/main' does
    not exist in this repository`` instead of exercising the stubbed
    ``list_git_tags``/``is_ancestor`` behaviour this class exists to check.

    The fourth read, ``_resolves_to_a_local_branch()``, is not stubbed: the
    failing-ancestry test reaches it, and it runs ``git symbolic-ref``
    against the real ``REPO_ROOT``. Both FAIL messages it can select between
    start with the text that test asserts on, so its result does not change
    the outcome.
    """

    def test_branch_containing_the_newest_tag_passes(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "assert_ref_is_qualified_and_exists", lambda _ref: None
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "list_git_tags", lambda: ["v0.5.0", "v0.6.0"]
        )
        monkeypatch.setattr(gate_module, "is_ancestor", lambda _tag, _branch: True)  # type: ignore[attr-defined]
        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        assert passed is True
        assert any(line.startswith("PASS") for line in messages)
        assert any("v0.6.0" in line for line in messages)

    def test_branch_missing_the_newest_tag_fails_and_names_both(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "assert_ref_is_qualified_and_exists", lambda _ref: None
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "list_git_tags", lambda: ["v0.5.0", "v0.6.0"]
        )
        monkeypatch.setattr(gate_module, "is_ancestor", lambda _tag, _branch: False)  # type: ignore[attr-defined]
        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "v0.6.0" in fail_lines[0]
        assert "refs/heads/main" in fail_lines[0]

    def test_no_matching_tags_raises(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "assert_ref_is_qualified_and_exists", lambda _ref: None
        )
        monkeypatch.setattr(gate_module, "list_git_tags", list)  # type: ignore[attr-defined]
        # Matched, not just raised: without the assert_ref_is_qualified_and_exists
        # stub above, "refs/heads/main" not existing in a given checkout
        # would also raise GateInputError here -- for the wrong reason, an
        # unstubbed existence check, never reaching newest_version_tag()'s
        # no-matching-tag path this test claims to cover. The match=
        # argument pins which raise this test is actually exercising.
        with pytest.raises(gate_module.GateInputError, match=r"no .* tag found"):  # type: ignore[attr-defined]
            gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]


class TestRunGateFailureRemediation:
    """A FAIL message must only prescribe an action the caller can actually take.

    Forward-merging 'dev' into the checked ref only makes sense when that
    ref is a local branch -- it is not possible to forward-merge into a
    tag, a remote-tracking ref, or a commit SHA, which are exactly the
    other shapes this gate accepts.
    """

    def test_a_local_branch_missing_the_tag_is_told_to_forward_merge(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "released", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        released_commit = _commit(repo, "the commit that gets tagged")
        subprocess.run(  # noqa: S603
            ["git", "tag", "v1.0.0", released_commit],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "feature", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "a diverging commit that never sees v1.0.0")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        passed, messages = gate_module.run_gate(branch="refs/heads/feature")  # type: ignore[attr-defined]
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "forward-merge" in fail_lines[0]

    def test_a_non_branch_ref_missing_the_tag_states_the_failure_without_a_remediation(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "tag", "checkpoint", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "the commit that gets the release tag")
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # "refs/tags/checkpoint" is a tag, not a branch -- there is nothing
        # to forward-merge into, so the message must not suggest it.
        passed, messages = gate_module.run_gate(branch="refs/tags/checkpoint")  # type: ignore[attr-defined]
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "forward-merge" not in fail_lines[0]

    def test_a_symbolic_ref_under_refs_heads_pointing_at_a_tag_is_not_a_local_branch(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        """A ``refs/heads/*`` ref can itself be symbolic and point outside ``refs/heads/``.

        Git accepts ``git symbolic-ref refs/heads/alias refs/tags/v1.0.0``
        without complaint, and resolves reads and writes through it. A
        prefix-only check would call ``refs/heads/alias`` a local branch
        and prescribe a forward merge that cannot actually be done --
        forwarding 'dev' into a tag is not an action anyone can take.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "tag", "v1.0.0", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "symbolic-ref", "refs/heads/alias", "refs/tags/v1.0.0"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "a diverging commit that never sees v1.0.0")
        subprocess.run(
            ["git", "tag", "v2.0.0"],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        passed, messages = gate_module.run_gate(branch="refs/heads/alias")  # type: ignore[attr-defined]
        assert passed is False
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "forward-merge" not in fail_lines[0]

    def test_a_symbolic_main_pointing_at_a_non_utf8_target_exits_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gate_module: object,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A symbolic ``refs/heads/main`` whose target is non-UTF-8 exits cleanly.

        ``_resolves_to_a_local_branch`` is only reached once ``contains_tag``
        is ``False`` (see :func:`run_gate`), so this needs ``main`` itself to
        be symbolic, its target to contain raw bytes that are not valid
        UTF-8, and the newest tag to *not* be reachable from it -- all three
        at once, built with ordinary git commands, no manual ``.git`` editing
        (``git clone --no-local --branch main`` yields a direct ``main``, not
        a symbolic one, which is why this state has to be constructed
        directly rather than reproduced through a clone).

        ``git symbolic-ref --quiet refs/heads/main`` succeeds (exit 0) and
        echoes the raw-byte target on stdout; decoding that eagerly as strict
        UTF-8 would raise ``UnicodeDecodeError`` -- outside
        :class:`GateInputError`, so it would escape ``main()`` as a bare
        traceback instead of the clean exit-1 diagnostic this module
        promises. This is exercised through ``main()`` with the default
        ``--branch refs/heads/main``, the invocation the release workflow
        uses.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        # The tag lives on a commit diverged from "base" -- unreachable from
        # "base" and therefore from "main" below -- so contains_tag is False
        # and run_gate() actually calls _resolves_to_a_local_branch().
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "released", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "the commit that gets tagged")
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)  # noqa: S607
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        # A branch name that is not valid UTF-8, the target "main" will
        # point through symbolically. Raw bytes 0x80 and 0xff can never
        # appear as a UTF-8 continuation or start byte in that position.
        raw_target_name = b"raw-" + bytes([0x80, 0xFF])
        subprocess.run(  # noqa: S603
            ["git", "branch", "--", raw_target_name],  # noqa: S607
            cwd=repo,
            check=True,
        )
        raw_target_ref = b"refs/heads/" + raw_target_name
        subprocess.run(  # noqa: S603
            ["git", "symbolic-ref", "refs/heads/main", raw_target_ref],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        exit_code = gate_module.main([])  # type: ignore[attr-defined]

        # main() returns the failure exit code instead of raising while it
        # resolves the FAIL remediation.
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.out


class TestIsAncestor:
    def test_raises_on_an_unresolvable_ref(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError):  # type: ignore[attr-defined]
            gate_module.is_ancestor(  # type: ignore[attr-defined]
                "refs/tags/this-tag-does-not-exist-anywhere", "HEAD"
            )


class TestIsAncestorAgainstADisposableRepository:
    """Exercise the real 0/1 exit-code path of ``git merge-base --is-ancestor``.

    This does not depend on this repository's own history (unlike
    ``TestFailabilityOnRealHistory`` below) -- it builds a throwaway
    two-branch repository under ``tmp_path`` instead, so it runs
    unconditionally, including on a shallow CI checkout.
    """

    def test_true_and_false_cases_are_told_apart(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(
            ["git", "checkout", "--quiet", "-b", "branch-a"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        commit_a = _commit(repo, "commit on branch-a")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "branch-b", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        commit_b = _commit(repo, "commit on branch-b")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # base is an ancestor of commit_b (branch-b was cut from it).
        assert gate_module.is_ancestor(base, commit_b) is True  # type: ignore[attr-defined]
        # commit_a and commit_b are on diverging branches -- neither is an
        # ancestor of the other, and both refs resolve cleanly, so this
        # exercises the real "resolves fine, exit code 1" case rather than
        # the unresolvable-ref error path.
        assert gate_module.is_ancestor(commit_a, commit_b) is False  # type: ignore[attr-defined]


class TestUnicodeTagSpoofingAgainstADisposableRepository:
    """Two Unicode assumptions the tag parser must not make.

    Both build a throwaway repository under ``tmp_path`` -- unconditional,
    like ``TestIsAncestorAgainstADisposableRepository`` above -- rather than
    depending on any tag actually present in this repository's own history.

    1. The numeric component of ``TAG_RE`` must match ASCII digits only.
       ``\\d`` in a Python regex is Unicode-aware, so it also matches
       non-ASCII decimal digits (e.g. U+0662 ARABIC-INDIC DIGIT TWO), and
       ``int()`` converts what it matches. A tag such as ``v1٩.0.0`` would
       parse as if its major component were 19, not as a non-match.
    2. ``parse_tag_version`` must not strip whitespace before matching.
       ``str.strip()`` removes Unicode whitespace, not just ASCII space, so
       a tag differing from a canonical one only by trailing U+00A0
       NO-BREAK SPACE would parse identically to the canonical spelling --
       and ``git tag`` accepts that trailing character
       (``git check-ref-format`` rejects ASCII space and control characters
       in a ref name, but not this).
    """

    def test_a_non_ascii_digit_tag_that_looks_newer_hides_an_unmerged_real_release(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        # 'git init' does not reliably create a branch named "main" (it
        # depends on 'init.defaultBranch'), so name one explicitly rather
        # than assume it, matching the pattern used elsewhere in this file.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 11")
        subprocess.run(["git", "tag", "v11.0.0"], cwd=repo, check=True)  # noqa: S607

        # The real v12.0.0 release lives on a branch that was never merged
        # into main -- exactly the gap this gate exists to catch.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 12, published but never forward-merged")
        subprocess.run(["git", "tag", "v12.0.0"], cwd=repo, check=True)  # noqa: S607

        # Back on main, a tag whose major component is an ASCII "1" followed
        # by U+0669 ARABIC-INDIC DIGIT NINE -- int("1٩") == 19, which
        # outranks both v11.0.0 and the real v12.0.0 above, and it *is*
        # reachable from main.
        subprocess.run(
            ["git", "checkout", "--quiet", "main"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "cosmetic commit on main")
        subprocess.run(
            ["git", "tag", "v1٩.0.0"],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        # If the Unicode-digit tag parsed as (19, 0, 0) it would outrank the
        # real v12.0.0 and, being reachable, make the gate report PASS while
        # the real v12.0.0 release sat unreachable from main.
        assert passed is False, messages
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "v12.0.0" in fail_lines[0]

    def test_a_tag_differing_only_by_trailing_unicode_whitespace_is_not_canonical(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 11")
        subprocess.run(["git", "tag", "v11.0.0"], cwd=repo, check=True)  # noqa: S607

        # The real v12.0.0 release lives on a branch that was never merged
        # into main -- the same shape as the digit-spoofing test above, but
        # this time the thing that must not shadow it is whitespace, not a
        # non-ASCII digit.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 12, published but never forward-merged")
        subprocess.run(["git", "tag", "v12.0.0"], cwd=repo, check=True)  # noqa: S607

        # Back on main, a tag that differs from 'v13.0.0' by one trailing
        # U+00A0 NO-BREAK SPACE. 13 > 12 uniquely (no tie with the real
        # v12.0.0 tag to be resolved by list order), so if this tag is
        # ever treated as canonical it strictly outranks the real release
        # and, being reachable, would hide that v12.0.0 never reached main.
        subprocess.run(
            ["git", "checkout", "--quiet", "main"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "cosmetic commit on main")
        subprocess.run(
            ["git", "tag", "v13.0.0 "],  # noqa: S607, RUF001 -- deliberate NBSP, see above
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        # 'v13.0.0\xa0'.strip() == 'v13.0.0', so if the parser stripped
        # whitespace this tag would parse as (13, 0, 0), uniquely outrank the
        # real v12.0.0 and, being reachable, make the gate report PASS while
        # the real v12.0.0 release sat unreachable from main. It must not
        # stand in for a canonical tag that was never actually created.
        assert passed is False, messages
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "v12.0.0" in fail_lines[0]


class TestTagAncestryUsesTheQualifiedTagRef:
    """The tag side of the ancestry check must be qualified, like the branch side.

    ``run_gate`` passes the tag to ``is_ancestor`` as ``refs/tags/<tag>``,
    not as the bare name ``list_git_tags`` returns (e.g. ``"v12.0.0"``).
    Git resolves a bare name through its usual
    search order (``$GIT_DIR/<name>``, then ``refs/<name>``, then
    ``refs/tags/<name>``, then ``refs/heads/<name>``, ...), so a same-named
    ref that sits earlier in that order can shadow the real tag entirely.
    ``assert_ref_is_qualified_and_exists`` protects ``branch`` against
    exactly this ambiguity, and ``run_gate`` qualifies the tag as
    ``refs/tags/<tag>`` to protect the other argument of the same call.

    This builds the shape that exposes the ambiguity: a genuine
    ``v12.0.0`` tag that is unreachable from ``main``, and a generic
    ``refs/v12.0.0`` ref (not a tag, not a branch -- a top-level ref under
    ``refs/``) that *is* reachable from ``main``. ``refs/<name>`` is tried before
    ``refs/tags/<name>`` in git's search order, so resolving the tag by its
    bare name would pick the generic ref instead of the real tag and the
    gate would report PASS while the real release sat unreachable. With the
    tag qualified as ``refs/tags/<tag>``, the generic ref is never consulted
    and the gate must report FAIL.
    """

    def test_a_generic_ref_sharing_the_tags_bare_name_must_not_shadow_the_real_tag(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 11")
        subprocess.run(["git", "tag", "v11.0.0"], cwd=repo, check=True)  # noqa: S607

        # The real v12.0.0 release lives on a branch that was never merged
        # into main -- exactly the gap this gate exists to catch.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 12, published but never forward-merged")
        subprocess.run(["git", "tag", "v12.0.0"], cwd=repo, check=True)  # noqa: S607

        # Back on main: a generic ref sharing the tag's bare name, pointing
        # at main's own tip -- reachable from main, unlike the real tag.
        # 'refs/v12.0.0' is neither 'refs/tags/v12.0.0' nor
        # 'refs/heads/v12.0.0'; git's bare-name search order tries
        # 'refs/<name>' before 'refs/tags/<name>', so an unqualified lookup
        # of "v12.0.0" resolves to this ref first.
        subprocess.run(["git", "checkout", "--quiet", "main"], cwd=repo, check=True)  # noqa: S607
        main_tip = _commit(repo, "cosmetic commit on main")
        subprocess.run(  # noqa: S603
            ["git", "update-ref", "refs/v12.0.0", main_tip],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        # If is_ancestor() received the bare "v12.0.0", git would resolve it
        # to 'refs/v12.0.0' (reachable from main) instead of the real
        # 'refs/tags/v12.0.0' (not reachable), and this would return
        # (True, ...) -- reporting the newest version tag as on main when it
        # is not.
        assert passed is False, messages
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "v12.0.0" in fail_lines[0]


class TestListGitTagsSplitsOnlyOnALiteralNewline:
    """``list_git_tags`` splits git's output on a literal newline only.

    ``str.splitlines()`` breaks on more than ``"\\n"`` -- it also treats
    U+0085 NEL (among other separators) as a line boundary, and git accepts
    U+0085 inside a tag name (see the comment inside ``list_git_tags``). It
    might seem that a fragment produced by such a split is harmless on its
    own: it would either fail ``TAG_RE`` or name a ref that does not exist
    and raise. That does not hold -- a fragment can still shadow a real,
    lower-numbered release by outranking it in ``newest_version_tag`` and
    then failing to resolve, which masks the correct tag behind a
    ``GateInputError`` instead of letting the gate run against it. This is
    independent of whether the tag side of ``is_ancestor`` is qualified:
    the fragment here never resolves to *anything* real, so qualifying it
    would only change the shape of the failure, not this test's ability to
    detect the split.
    """

    def test_a_tag_fractured_by_splitlines_does_not_shadow_the_real_tag(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 11")
        subprocess.run(["git", "tag", "v11.0.0"], cwd=repo, check=True)  # noqa: S607
        _commit(repo, "cosmetic commit on main")
        # A tag name containing U+0085 NEL. str.splitlines() treats this as
        # a line boundary and would fracture git's single output line for
        # this tag into two bogus entries, "v13.0.0" and "junk" --
        # neither of which is a real ref in this repository. "v13.0.0" is
        # shaped like a valid version tag and would outrank the real
        # v11.0.0 in newest_version_tag() if it were ever treated as a
        # candidate.
        subprocess.run(["git", "tag", "v13.0.0\x85junk"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # Splitting on a literal "\n" only, git's output for this tag comes
        # back as one line containing the NEL character, which does not
        # match TAG_RE -- so it is correctly excluded, and the gate runs
        # against the one real tag, v11.0.0, which is reachable from main.
        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        assert passed is True, messages
        assert any("v11.0.0" in line for line in messages)


class TestListGitTagsIsNotSubjectToPorcelainDisplayConfiguration:
    """``list_git_tags`` must not read a porcelain command's formatted output.

    ``git tag -l`` is porcelain -- it formats its output for a human reader
    and honours display configuration that has nothing to do with which
    tags exist. With ``column.tag=always`` set and a narrow terminal width
    (``COLUMNS=20``), ``git tag -l 'v*'`` against a repository with
    reachable tags up to ``v3.0.0`` and unreachable ``v4.0.0`` and
    ``v5.0.0`` prints::

        v1.0.0  v4.0.0
        v2.0.0  v5.0.0
        v3.0.0

    Each of the first two lines holds two tag names, so neither matches
    ``TAG_RE`` and both would be discarded by :func:`newest_version_tag`.
    That would leave only ``v3.0.0``, which is reachable, so the gate would
    report PASS while the real newest release, ``v5.0.0``, sits unreachable
    from ``main``. ``list_git_tags`` reads ``git for-each-ref``, a plumbing
    command that is not subject to ``column.*`` configuration (see the
    function's docstring), so this must report FAIL.
    """

    def test_column_tag_always_with_a_narrow_terminal_does_not_hide_the_newest_tag(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        for name in ("v1.0.0", "v2.0.0", "v3.0.0"):
            _commit(repo, f"release {name}")
            subprocess.run(["git", "tag", name], cwd=repo, check=True)  # noqa: S603, S607

        # v4.0.0 and v5.0.0 are published on a branch that is never
        # forward-merged into main -- exactly the gap this gate exists to
        # catch. They must outrank v1.0.0-v3.0.0 so that hiding them behind
        # the porcelain column fracture would make the gate wrongly pass.
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "side", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        for name in ("v4.0.0", "v5.0.0"):
            _commit(repo, f"release {name}, never forward-merged")
            subprocess.run(["git", "tag", name], cwd=repo, check=True)  # noqa: S603, S607

        subprocess.run(["git", "checkout", "--quiet", "main"], cwd=repo, check=True)  # noqa: S607

        # column.tag=always forces column formatting even though this
        # subprocess is never attached to a terminal.
        subprocess.run(
            ["git", "config", "column.tag", "always"],  # noqa: S607
            cwd=repo,
            check=True,
        )
        # A narrow width is what packs two six-character tag names onto one
        # line; git falls back to this environment variable when stdout is
        # not a terminal it can query for a real width.
        monkeypatch.setenv("COLUMNS", "20")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        # Were list_git_tags() to read 'git tag -l's columnised output, only
        # 'v3.0.0' would survive TAG_RE, and it is reachable from main --
        # so this would return (True, ...) while the real newest release,
        # v5.0.0, is not on main at all.
        assert passed is False, messages
        assert any("v5.0.0" in line for line in messages)
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        assert "v5.0.0" in fail_lines[0]


class TestListGitTagsToleratesANonUtf8TagName:
    """A non-UTF-8 tag name must be ignored, not crash the process.

    Git accepts tag names containing raw bytes ``0x80``-``0xff`` that are
    not valid UTF-8 (the test creates one below). Decoding ``git``'s output
    as strict UTF-8 would raise an uncaught ``UnicodeDecodeError`` the moment
    such a tag existed -- a traceback, not a clean exit-1 diagnostic.
    ``list_git_tags`` decodes with ``errors="surrogateescape"`` instead, so
    the name reaches ``TAG_RE``, fails to match it like any other non-canonical
    name, and is silently excluded -- exactly like a prerelease suffix or
    any other tag that is not a canonical ``vMAJOR.MINOR.PATCH`` name.
    """

    def test_a_non_utf8_tag_is_ignored_rather_than_crashing(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        _commit(repo, "release 1")
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)  # noqa: S607
        _commit(repo, "cosmetic commit on main")
        # A tag name that is not valid UTF-8: raw bytes 0x80 and 0xff can
        # never appear as a UTF-8 continuation or start byte in that
        # position, so this cannot be decoded strictly no matter which
        # Unicode codepoints were intended.
        raw_tag_name = b"v" + bytes([0x80, 0xFF]) + b".0.0"
        subprocess.run(  # noqa: S603
            ["git", "tag", "--", raw_tag_name],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # The call must run to completion, ignore the non-UTF-8 tag as
        # non-canonical, and report the one real tag; a strict UTF-8 decode
        # would raise UnicodeDecodeError here instead.
        passed, messages = gate_module.run_gate(branch="refs/heads/main")  # type: ignore[attr-defined]
        assert passed is True, messages
        assert any("v1.0.0" in line for line in messages)


class TestIsWellFormedRefPathToleratesNonUtf8Bytes:
    """A ``--branch`` value with raw invalid-UTF-8 bytes must not crash.

    When the OS hands Python an ``argv`` entry it cannot decode as UTF-8,
    CPython decodes it anyway with ``errors="surrogateescape"`` (PEP 383):
    the value keeps every original byte, just held as an unpaired
    surrogate per undecodable byte. ``refs/heads/\\udcff\\udcfe`` is exactly
    that shape for the two raw bytes ``0xff`` and ``0xfe``. Passed to ``git
    check-ref-format --normalize`` as a subprocess argument, that surrogate
    round-trips back to the identical raw bytes on the way out (the same
    codec is how ``os.fsencode``/``exec`` handle ``argv`` on POSIX), so git
    receives exactly what the OS would have handed a C program directly --
    and git accepts this as a syntactically well-formed (if nonexistent)
    ref path.

    ``_is_well_formed_ref_path`` runs that subprocess call without
    ``text=True``, which would decode *both* streams regardless of whether
    their content is ever used -- and it is not used here, only the exit
    code is. ``--normalize`` echoes the (still byte-laden) ref back to
    stdout on success; capturing raw bytes means nothing decodes it.
    """

    def test_a_branch_with_non_utf8_bytes_exits_cleanly_instead_of_crashing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gate_module: object,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "initial commit")
        subprocess.run(["git", "branch", "main"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        branch = "refs/heads/" + "\udcff\udcfe"

        exit_code = gate_module.main(["--branch", branch])  # type: ignore[attr-defined]

        # main() returns a clean exit code; a strict UTF-8 decode in
        # _is_well_formed_ref_path would raise UnicodeDecodeError from the
        # call above instead.
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err


class TestIsAncestorToleratesNonUtf8Bytes:
    """is_ancestor's failure-to-resolve message must not crash, on decode or on write.

    ``run_gate()`` calls ``assert_ref_is_qualified_and_exists(branch)``
    before ``is_ancestor()``, and that guard rejects a raw-byte branch
    value that does not resolve to an existing object (see
    ``TestIsWellFormedRefPathToleratesNonUtf8Bytes``, where the same shape
    of value fails with a "does not exist" diagnostic before ``is_ancestor``
    runs). ``is_ancestor`` is still reachable from an unmodified ``main()``
    with a raw-byte name, through an existing ref that resolves to a blob
    rather than a commit: ``git merge-base --is-ancestor`` then exits 128
    and echoes that raw name on stderr. ``assert_ref_is_qualified_and_exists``
    is monkeypatched to a no-op here so the test reaches ``is_ancestor``
    through ``main()`` with a name git cannot resolve, rather than calling
    ``is_ancestor`` as a bare unit: the requirement is that ``main()``
    never lets an exception escape on this path, not merely that the helper
    returns a value.

    ``capsys`` is requested deliberately: it replaces ``sys.stderr`` with a
    stream using the strict ``"utf-8"`` codec, so an unsanitised write of a
    surrogate-laden message would raise ``UnicodeEncodeError`` here.
    """

    def test_a_branch_with_non_utf8_bytes_exits_cleanly_instead_of_crashing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gate_module: object,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            gate_module, "assert_ref_is_qualified_and_exists", lambda _ref: None
        )

        branch = "refs/heads/" + "\udcff\udcfe"

        exit_code = gate_module.main(["--branch", branch])  # type: ignore[attr-defined]

        # main() must return: a strict decode inside is_ancestor's
        # subprocess call, or the write of its surrogate-laden message to a
        # strict sys.stderr, would each raise instead.
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "could not resolve one of the refs" in captured.err


class TestListGitTagsErrorMessageToleratesNonUtf8Bytes:
    """list_git_tags' own failure message must not crash on write either.

    This test fakes the one ``for-each-ref`` subprocess call to fail with a
    non-UTF-8 ``fatal:`` message, routing every other git invocation
    ``run_gate()`` makes (``check-ref-format``, ``rev-parse``,
    ``merge-base``) through the real ``subprocess.run`` unchanged.
    ``list_git_tags`` decodes this stderr with ``errors="surrogateescape"``
    instead of crashing on decode; the remaining concern is the later write
    of the decoded message to ``sys.stderr``, the same one ``is_ancestor``'s
    message has (see ``TestIsAncestorToleratesNonUtf8Bytes``).
    """

    def test_a_non_utf8_for_each_ref_failure_exits_cleanly_instead_of_crashing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gate_module: object,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "initial commit")
        subprocess.run(["git", "branch", "main"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        real_run = gate_module.subprocess.run  # type: ignore[attr-defined]

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            if args[:2] == ["git", "for-each-ref"]:
                return subprocess.CompletedProcess(
                    args,
                    returncode=1,
                    stdout=b"",
                    stderr=b"fatal: bad pattern \xff\xfe\n",
                )
            return real_run(args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(gate_module.subprocess, "run", fake_run)  # type: ignore[attr-defined]

        exit_code = gate_module.main(["--branch", "refs/heads/main"])  # type: ignore[attr-defined]

        # main() sanitises this message before writing it; an unsanitised
        # write to capsys' strict stderr would raise UnicodeEncodeError
        # instead of returning.
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "for-each-ref" in captured.err


class TestSuccessReportToleratesNonUtf8BranchBytes:
    """The success-path report line must not crash on write either.

    Besides the write inside ``main()``'s ``except GateInputError`` branch
    (see ``TestIsAncestorToleratesNonUtf8Bytes`` and
    ``TestListGitTagsErrorMessageToleratesNonUtf8Bytes`` above), the
    success path -- reached once validation *and* ancestry both succeed --
    interpolates the checked ``branch`` into a "branch checked: ..." report
    line and writes it to ``sys.stdout``. An *existing* branch whose name
    contains raw bytes that are not valid UTF-8 (accepted by git; the test
    creates one below) can reach that write as text carrying unpaired
    surrogates, and a strict stream raises
    ``UnicodeEncodeError`` on such text unless it is sanitised first --
    not during validation, not during the ancestry check, only while
    printing the result that both already succeeded.

    ``capsys`` provides the strict stream this needs: it installs a
    strict-UTF-8 ``sys.stdout``, as ``TestIsAncestorToleratesNonUtf8Bytes``
    relies on for ``sys.stderr``.
    """

    def test_an_existing_branch_with_non_utf8_bytes_prints_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gate_module: object,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(["git", "tag", "v1.0.0", base], cwd=repo, check=True)  # noqa: S603, S607
        # A branch name that is not valid UTF-8: raw bytes 0x80 and 0xff can
        # never appear as a UTF-8 continuation or start byte in that
        # position. Created with a raw bytes argv entry (not a Python str)
        # so git receives the exact bytes -- the same technique
        # TestListGitTagsToleratesANonUtf8TagName uses for a tag name.
        raw_branch_name = b"branch-" + bytes([0x80, 0xFF])
        subprocess.run(  # noqa: S603
            ["git", "branch", "--", raw_branch_name],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # The Python-str spelling of the same raw bytes, as CPython would
        # decode an undecodable argv entry (PEP 383, surrogateescape): the
        # two invalid bytes become the unpaired surrogates \udc80 and \udcff.
        branch = "refs/heads/branch-" + "\udc80\udcff"

        exit_code = gate_module.main(["--branch", branch])  # type: ignore[attr-defined]

        # main() returns 0; an unsanitised success-path write of
        # "branch checked: <branch>" would raise UnicodeEncodeError instead.
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "branch checked" in captured.out


class TestReportWriteToleratesTheActualStreamEncoding:
    """A report write of a non-ASCII ref must not raise on a strict ASCII ``sys.stdout``.

    ``_safe_for_stream`` (the sanitiser every write in ``main()`` goes
    through) re-encodes against ``stream.encoding`` rather than a hardcoded
    ``"utf-8"``. A sanitiser that only guaranteed "the result is UTF-8
    encodable" would not guarantee "safe to write to the stream actually
    installed as ``sys.stdout``/``sys.stderr``": a perfectly legitimate,
    valid-UTF-8 ref such as ``refs/heads/é`` passes such a sanitiser
    unchanged (nothing about it is unencodable as UTF-8), yet writing it to
    a strict ASCII stream raises ``UnicodeEncodeError``.

    This substitutes ``sys.stdout`` directly with a ``TextIOWrapper`` around
    an in-memory buffer, configured with ``encoding="ascii", errors="strict"``.
    """

    def test_a_legitimate_unicode_branch_prints_cleanly_on_a_strict_ascii_stream(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "base commit")
        subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)  # noqa: S607
        subprocess.run(["git", "branch", "é"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        buffer = io.BytesIO()
        strict_ascii_stdout = io.TextIOWrapper(buffer, encoding="ascii", errors="strict")
        monkeypatch.setattr("sys.stdout", strict_ascii_stdout)

        # The success-path write of "branch checked: refs/heads/é" must not
        # raise UnicodeEncodeError even though "é" is valid UTF-8: the
        # sanitiser has to handle the ASCII sys.stdout installed above, not
        # just UTF-8 encodability.
        exit_code = gate_module.main(["--branch", "refs/heads/é"])  # type: ignore[attr-defined]

        strict_ascii_stdout.flush()
        assert exit_code == 0
        assert b"branch checked" in buffer.getvalue()


class TestSafeForStreamDoesNotCoverAnUnregisteredCodecName:
    """Pins that ``_safe_for_stream`` raises ``LookupError`` for an unregistered codec name.

    A stream whose ``.encoding`` names a codec the codec registry does not
    recognise (``"x-no-such-codec"``) makes ``_safe_for_stream`` raise
    ``LookupError`` rather than returning a sanitised string. This is
    deliberately *not* treated as a defect: an ``io.TextIOWrapper``
    constructed with such a name raises ``LookupError`` in its own
    constructor.

    This test exists so a future change that adds a silent fallback for
    this case fails loudly here instead of shipping unnoticed.
    """

    def test_an_unregistered_codec_name_raises_lookuperror_rather_than_falling_back(
        self, gate_module: object
    ) -> None:
        class _FakeStreamWithBogusEncoding:
            encoding = "x-no-such-codec"

        with pytest.raises(LookupError):
            gate_module._safe_for_stream(  # type: ignore[attr-defined]
                "hello", _FakeStreamWithBogusEncoding()
            )


class TestArgparseWritesGoThroughTheSanitisingChokePoint:
    """argparse's own help/usage/error writes must not bypass ``_write``.

    The base ``ArgumentParser`` writes directly to ``sys.stdout``
    (``--help``) or ``sys.stderr`` (an unrecognised argument) without
    calling :func:`_write`, while ``_write``'s own docstring promises every
    write ``main()`` makes goes through it. ``main()`` therefore uses a
    parser subclass that routes those writes through ``_write``. The first
    test below pins that the choke point is actually used. The second pins
    the concrete failure a bypass would cause: with a strict-UTF-8
    ``sys.stderr`` installed (the same technique
    ``TestReportWriteToleratesTheActualStreamEncoding`` uses for
    ``sys.stdout``), the base parser raises ``UnicodeEncodeError`` from its
    own unsanitised ``file.write(message)`` for ``main(["bad-\\udcff"])`` --
    after printing only the usage line, never reaching the "unrecognized
    arguments" message that names the bad value.
    """

    def test_help_output_is_routed_through_write(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        calls: list[str] = []
        original_write = gate_module._write  # type: ignore[attr-defined]

        def recording_write(stream: object, text: str) -> None:
            calls.append(text)
            original_write(stream, text)  # type: ignore[operator]

        monkeypatch.setattr(gate_module, "_write", recording_write)  # type: ignore[attr-defined]

        with pytest.raises(SystemExit) as exc_info:
            gate_module.main(["--help"])  # type: ignore[attr-defined]

        assert exc_info.value.code == 0
        # The base ArgumentParser's print_help() writes straight to
        # sys.stdout without calling _write(), which would leave this list
        # empty.
        assert calls, "expected --help output to go through _write()"
        assert any("usage:" in call for call in calls)

    def test_unrecognized_argument_error_survives_a_strict_stream(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object
    ) -> None:
        buffer = io.BytesIO()
        strict_stderr = io.TextIOWrapper(buffer, encoding="utf-8", errors="strict")
        monkeypatch.setattr("sys.stderr", strict_stderr)

        # A positional argument (this parser defines none) containing an
        # unpaired surrogate -- the shape CPython hands back for an argv
        # entry the OS could not decode as UTF-8 (PEP 383). The base
        # ArgumentParser would raise UnicodeEncodeError here instead of
        # exiting cleanly.
        with pytest.raises(SystemExit) as exc_info:
            gate_module.main(["bad-\udcff"])  # type: ignore[attr-defined]

        assert exc_info.value.code == 2
        strict_stderr.flush()
        written = buffer.getvalue()
        assert b"usage:" in written
        assert b"unrecognized arguments" in written


class TestAssertRefIsQualifiedAndExists:
    """Only a fully qualified ref path or a full object ID is accepted -- nothing else.

    Git resolves a bare name through a search order across several ref
    namespaces, and also as an abbreviated object ID, so a guard that
    enumerated those candidates and refused a bare name only on a collision
    would have to track that search order. This class pins the alternative:
    bare names, 'HEAD', abbreviated SHAs, and revision expressions are
    refused unconditionally.
    """

    def test_bare_name_is_refused(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("main")  # type: ignore[attr-defined]

    def test_head_is_refused(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("HEAD")  # type: ignore[attr-defined]

    def test_abbreviated_sha_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        commit = _commit(repo, "initial commit")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists(commit[:8])  # type: ignore[attr-defined]

    def test_revision_expression_on_a_bare_name_is_refused(self, gate_module: object) -> None:
        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("main^0")  # type: ignore[attr-defined]

    def test_revision_expression_on_a_qualified_ref_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        """A revision expression on a qualified ref path is refused.

        ``refs/heads/main^0`` starts with 'refs/', but names a ref plus a
        suffix operation, not the ref itself, so it must not be accepted as
        a qualified ref path.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "initial commit")
        subprocess.run(["git", "branch", "main"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("refs/heads/main^0")  # type: ignore[attr-defined]

    def test_qualified_ref_that_exists_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "initial commit")
        subprocess.run(["git", "branch", "release"], cwd=repo, check=True)  # noqa: S607

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # Must not raise.
        gate_module.assert_ref_is_qualified_and_exists("refs/heads/release")  # type: ignore[attr-defined]

    def test_qualified_ref_that_does_not_exist_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "initial commit")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # Well-formed but nonexistent: must fail with a message naming it,
        # not a traceback further down the call chain.
        with pytest.raises(gate_module.GateInputError, match="refs/heads/nope"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("refs/heads/nope")  # type: ignore[attr-defined]

    def test_full_object_id_that_exists_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        commit = _commit(repo, "initial commit")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        assert len(commit) == 40
        # Must not raise.
        gate_module.assert_ref_is_qualified_and_exists(commit)  # type: ignore[attr-defined]

    def test_uppercase_full_object_id_that_exists_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        commit = _commit(repo, "initial commit")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        uppercase = commit.upper()
        assert uppercase != commit
        # Git resolves the uppercase spelling of a full object ID exactly
        # like the lowercase one -- must not raise.
        gate_module.assert_ref_is_qualified_and_exists(uppercase)  # type: ignore[attr-defined]

    def test_full_object_id_that_does_not_exist_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        _commit(repo, "initial commit")

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        nonexistent = "f" * 40
        with pytest.raises(gate_module.GateInputError, match=nonexistent):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists(nonexistent)  # type: ignore[attr-defined]

    def test_root_pseudo_ref_colliding_with_a_branch_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        """A root pseudo-ref that collides with a branch is refused.

        Git tries ``$GIT_DIR/<name>`` for any name, not just a handful of
        well-known ones. Here 'FOO' is a root pseudo-ref pointing at the
        tagged commit while 'refs/heads/FOO' points at an earlier one; git
        itself warns this is ambiguous. There is no list of candidates to
        be incomplete: 'FOO' is a bare name and is refused outright,
        independent of what it collides with.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        earlier = _commit(repo, "earlier commit")
        subprocess.run(  # noqa: S603
            ["git", "branch", "FOO", earlier],  # noqa: S607
            cwd=repo,
            check=True,
        )
        tagged = _commit(repo, "the commit that gets tagged")
        subprocess.run(  # noqa: S603
            ["git", "tag", "v1.0.0", tagged],  # noqa: S607
            cwd=repo,
            check=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "update-ref", "FOO", tagged],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("FOO")  # type: ignore[attr-defined]

    def test_abbreviated_object_id_colliding_with_a_branch_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        """An abbreviated commit ID that also names a branch is refused.

        An abbreviation is neither a full object ID nor a qualified ref
        path, so it is refused outright, independent of what it collides
        with.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        commit = _commit(repo, "initial commit")
        abbreviated = commit[:8]
        subprocess.run(  # noqa: S603
            ["git", "branch", abbreviated],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists(abbreviated)  # type: ignore[attr-defined]

    def test_origin_slash_main_is_refused(self, gate_module: object) -> None:
        """A slash-containing bare name must not be mistaken for a qualified ref path.

        ``origin/main`` is exactly the shape a caller is most likely to
        pass by mistake instead of ``refs/remotes/origin/main`` -- it
        contains a slash and reads like a ref, but it does not start with
        ``refs/`` and git would resolve it by search order like any other
        bare name. This must be refused unconditionally, independent of
        this repository's own history.
        """
        with pytest.raises(gate_module.GateInputError, match="not an accepted ref"):  # type: ignore[attr-defined]
            gate_module.assert_ref_is_qualified_and_exists("origin/main")  # type: ignore[attr-defined]

    def test_qualified_ref_is_accepted_despite_a_branch_and_tag_sharing_its_bare_name(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        """A qualified ref path must resolve to itself, not to a same-named sibling.

        A branch and a tag named ``release`` point at different commits
        here. ``refs/heads/release`` is unambiguous by construction, but a
        regression that resolved by bare name somewhere along the way could
        silently pick the tag instead. Checked via ``is_ancestor`` (which
        receives the qualified ref exactly as constructed) rather than mere
        existence, so that kind of substitution would show up as a wrong
        answer, not just a wrongly-skipped raise.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "release", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        commit_on_branch = _commit(repo, "commit on the release branch")
        subprocess.run(  # noqa: S603
            ["git", "checkout", "--quiet", "-b", "other", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        commit_on_tag = _commit(repo, "a diverging commit that the tag points at")
        subprocess.run(  # noqa: S603
            ["git", "tag", "release", commit_on_tag],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # Must not raise -- the branch and the tag coexist under the same
        # bare name, but the qualified path names one of them exactly.
        gate_module.assert_ref_is_qualified_and_exists("refs/heads/release")  # type: ignore[attr-defined]
        # It must resolve to the branch tip, not the diverging tag commit.
        assert gate_module.is_ancestor(commit_on_branch, "refs/heads/release") is True  # type: ignore[attr-defined]
        assert gate_module.is_ancestor(commit_on_tag, "refs/heads/release") is False  # type: ignore[attr-defined]

    def test_refs_remotes_origin_main_is_accepted_and_resolves_correctly(
        self, monkeypatch: pytest.MonkeyPatch, gate_module: object, tmp_path: Path
    ) -> None:
        """``refs/remotes/origin/main`` is accepted unconditionally and resolves to itself.

        This shape is also exercised inside ``TestFailabilityOnRealHistory``,
        which is skipped when the required history is absent from the
        checkout. This test is built on a disposable repository so it runs
        unconditionally. A local branch named ``main`` exists at a different
        commit to prove this resolves the remote-tracking ref itself, not
        the local branch of the same bare name.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_disposable_repo(repo)
        base = _commit(repo, "base commit")
        subprocess.run(  # noqa: S603
            ["git", "branch", "main", base],  # noqa: S607
            cwd=repo,
            check=True,
        )
        remote_main_commit = _commit(repo, "commit that only origin/main carries")
        subprocess.run(  # noqa: S603
            ["git", "update-ref", "refs/remotes/origin/main", remote_main_commit],  # noqa: S607
            cwd=repo,
            check=True,
        )

        monkeypatch.setattr(gate_module, "REPO_ROOT", repo)  # type: ignore[attr-defined]

        # Must not raise.
        gate_module.assert_ref_is_qualified_and_exists("refs/remotes/origin/main")  # type: ignore[attr-defined]
        assert (  # type: ignore[attr-defined]
            gate_module.is_ancestor(remote_main_commit, "refs/remotes/origin/main") is True
        )
        # The local branch "main" sits at "base", which is an ancestor of
        # everything here -- so the discriminating check is the reverse:
        # "base" being an ancestor does not prove which ref was resolved,
        # but "remote_main_commit" only exists on origin/main, so its
        # presence as an ancestor of the resolved ref (checked above) is
        # only possible if the remote-tracking ref -- not local "main" --
        # was the one actually resolved.
        assert gate_module.is_ancestor(base, "refs/heads/main") is True  # type: ignore[attr-defined]
        assert gate_module.is_ancestor(remote_main_commit, "refs/heads/main") is False  # type: ignore[attr-defined]


@REQUIRES_V0_6_0
@REQUIRES_ORIGIN_MAIN
@REQUIRES_PRE_FORWARD_MERGE_COMMIT
class TestFailabilityOnRealHistory:
    """The check run against this repository's real history.

    Green: ``origin/main`` contains the newest version tag, so the gate
    passes.

    Red: the merge commit that forward-merged v0.6.0 into main is
    8c044e2 ("Merge pull request #49 from
    sovantica/chore/forward-merge-the-0-6-0-release"); its first parent,
    pinned above as PRE_FORWARD_MERGE_COMMIT, is the state of main just
    before that forward merge landed, and does not contain v0.6.0.
    Checking the tool against that exact commit is not a synthetic
    scenario -- it is the window this check exists to close, captured as
    real history.
    """

    def test_origin_main_today_contains_the_newest_version_tag(self, gate_module: object) -> None:
        passed, messages = gate_module.run_gate(branch="refs/remotes/origin/main")  # type: ignore[attr-defined]
        assert passed is True, messages
        assert any(line.startswith("PASS") for line in messages)

    def test_main_before_the_v0_6_0_forward_merge_does_not_contain_it(
        self, gate_module: object
    ) -> None:
        passed, messages = gate_module.run_gate(branch=PRE_FORWARD_MERGE_COMMIT)  # type: ignore[attr-defined]
        assert passed is False, messages
        fail_lines = [line for line in messages if line.startswith("FAIL")]
        assert fail_lines, messages
        # The gate names the newest version tag. That was v0.6.0 until a later
        # release was tagged; the commit contains neither v0.6.0 nor any later tag.
        newest, version = _newest_release_tag_in_this_repo()
        assert version >= (0, 6, 0)
        assert fail_lines[0].startswith(f"FAIL: {newest!r} is not reachable from ")
        assert PRE_FORWARD_MERGE_COMMIT in fail_lines[0]
