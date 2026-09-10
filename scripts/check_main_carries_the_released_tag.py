"""Check that ``main`` actually contains the newest ``v*`` version tag in this repository.

``BRANCHING.md`` states an invariant: after each release, ``main`` is
forward-merged from ``dev`` so the tagged release commit is reachable from
both branches. The release pipeline (``.github/workflows/release.yml``)
publishes to PyPI from ``dev`` and never touches ``main`` — the forward
merge is a separate, manual-or-scripted step that happens afterwards, and
nothing in the pipeline enforces that it ever happens. If it is forgotten,
fails, or is deferred, ``main`` silently falls behind what PyPI ships, and
the only way anyone notices today is a human comparing the two by hand.

This script automates the reachability half of that comparison: it resolves
the newest ``v*`` tag in this repository by semantic-version order and fails
if that tag is not an ancestor of the branch being checked
(``refs/heads/main`` by default). It is intentionally not wired into the
release job — the gap it is closing is the one that opens *after* a release
finishes, so it belongs on a schedule and on pushes to the branch it is
protecting, not on the publish path itself.

What a PASS from this script establishes, and only this: the newest
``vMAJOR.MINOR.PATCH`` tag that exists in this repository is reachable
(an ancestor, in git's sense) from the ref that was checked. The script
never contacts PyPI and never compares trees, so a PASS is not evidence
that the PyPI upload for that tag succeeded or finished, that the checked
ref's tree matches the tagged commit's tree, that some newer version
already on PyPI lacks a tag here (because it was never created or was
later deleted), or that the tag still points at the commit it pointed at
when the release published. It answers one narrow question — "is the tag
reachable from this ref?" — and nothing about the state of PyPI itself.

Usage::

    python scripts/check_main_carries_the_released_tag.py
    python scripts/check_main_carries_the_released_tag.py --branch refs/remotes/origin/main

Only two shapes of ``--branch`` are accepted: a fully qualified ref path
beginning with ``refs/`` (e.g. ``refs/heads/main``, ``refs/remotes/origin/main``,
``refs/tags/v0.6.0``), or a full 40-character hexadecimal object ID. A bare
name, ``HEAD``, an abbreviated SHA, and a revision expression such as
``main^0`` are all refused — see :func:`assert_ref_is_qualified_and_exists`.

Locally, fetch first and check ``refs/remotes/origin/main`` rather than the
default: a local ``refs/heads/main`` is typically stale relative to the
remote (a checkout of this repository can easily be over a hundred commits
behind). In CI, the checkout step points ``refs/heads/main`` at the same
commit the remote has, so the default is correct there without an extra step.

Exit codes:
* ``0`` — the newest version tag is an ancestor of the checked branch.
* ``1`` — it is not, or an input (branch, tag) could not be resolved.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import TextIO

    from _typeshed import SupportsWrite

REPO_ROOT = Path(__file__).resolve().parent.parent

# ``\d`` is Unicode-aware and matches far more than the ten ASCII digits
# (e.g. U+0662 ARABIC-INDIC DIGIT TWO), and ``int()`` accepts what it
# matches -- a tag such as ``v1٢.0.0`` would parse as 12. The class is
# spelled out explicitly, as ``_FULL_OBJECT_ID_RE`` below already does for
# the same reason, rather than compiling with ``re.ASCII``: that flag would
# apply to this whole pattern, and to any future one added to this module,
# so a later author adding e.g. ``\s`` to a new pattern here would silently
# inherit ASCII-only matching without a local reason to expect it.
_NUMERIC_COMPONENT = r"(?:0|[1-9][0-9]*)"
TAG_RE = re.compile(rf"^v({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})\.({_NUMERIC_COMPONENT})$")

EXIT_OK = 0
EXIT_FAIL = 1


class GateInputError(RuntimeError):
    """Raised when a branch, tag, or tag list cannot be resolved."""


def list_git_tags() -> list[str]:
    """Return every tag name in this repository, in no particular order.

    This shells out to ``git for-each-ref``, a plumbing command, rather than
    ``git tag -l``, a porcelain one -- confirmed by execution, not by
    reading the manual. ``git tag -l`` formats its output for a human
    reader and honours display configuration that has nothing to do with
    what tags exist: with ``column.tag=always`` set and a narrow terminal
    width, ``git tag -l 'v*'`` packs multiple tag names onto one line
    (e.g. ``"v1.0.0  v4.0.0"``), which does not match ``TAG_RE`` and is
    silently dropped by :func:`newest_version_tag` -- so a newer, unreachable
    tag can vanish from the candidate list entirely while an older,
    reachable one is reported as the newest, and the gate passes when it
    should not. ``--no-column`` closes that one report; it was rejected in
    favour of ``for-each-ref`` because column formatting is a property of
    porcelain commands in general, not a single flag on this one, and a
    reviewer already spent three rounds on this module discovering that
    enumerating porcelain behaviour instead of avoiding it keeps being
    incomplete (see :func:`assert_ref_is_qualified_and_exists`). Executed
    checks against git 2.51.0: ``for-each-ref`` output is identical with and
    without ``column.tag=always``/``column.ui=always`` at ``COLUMNS=20``;
    ``tag.sort`` only changes ordering, which this function does not rely
    on (:func:`newest_version_tag` computes the maximum itself); forcing
    ``pager.for-each-ref=true`` with ``core.pager=cat`` does not alter the
    bytes received by this subprocess either way, for ``for-each-ref`` or
    for ``tag -l``, because ``capture_output`` never attaches a pty for the
    pager to detect. The resulting tag set is *not* always identical to
    ``git tag -l 'v*'``'s, though -- confirmed by execution, in a repository
    with the tags ``v1.0.0``, ``v2.0.0``, ``v3/nested``, and
    ``vprefix/refs/tags/v999.0.0``, ``git tag -l 'v*'`` lists all four, but
    this call lists only ``v1.0.0`` and ``v2.0.0``. The two hierarchical
    names are excluded here because for-each-ref's pattern matches whole
    path components: its ``*`` does not cross a ``/`` boundary, so
    ``refs/tags/v*`` cannot match ``refs/tags/v3/nested`` or
    ``refs/tags/vprefix/refs/tags/v999.0.0`` -- querying
    ``refs/tags/v*/*`` instead does match ``v3/nested`` (also confirmed by
    execution), showing that the missing ``/``-crossing, not
    ``%(refname:lstrip=2)``'s formatting, is what excludes them.
    ``git tag -l``'s glob has no such restriction and matches ``v*`` against
    the bare tag name directly, letting ``*`` cross ``/`` freely.

    This never changes which tag :func:`newest_version_tag` selects, on
    either listing: ``TAG_RE`` anchors on ``^vMAJOR.MINOR.PATCH$`` with no
    ``/`` anywhere in the pattern, so a hierarchical name can never match it
    regardless of which of these two commands produced it. The two listings
    disagree about what exists under ``refs/tags/v*`` in a repository with
    hierarchical tag names; they cannot disagree about which one is the
    newest canonical version tag.
    """
    completed = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:lstrip=2)", "refs/tags/v*"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="surrogateescape")
        msg = f"'git for-each-ref' failed: {stderr.strip()}"
        raise GateInputError(msg)
    # Decoded with 'surrogateescape' rather than the strict default: git
    # accepts tag names containing raw bytes that are not valid UTF-8
    # (confirmed by creating one), and a strict decode would raise
    # UnicodeDecodeError before such a name ever reaches parse_tag_version,
    # crashing this script instead of the clean exit-1 diagnostic the
    # module promises for a bad input. 'surrogateescape' preserves every
    # byte losslessly as an unpaired surrogate, so a non-UTF-8 tag name
    # simply fails to match TAG_RE below like any other non-canonical name
    # -- it is never selected, and it never crashes the process.
    #
    # str.splitlines() is deliberately not used here either: it breaks on
    # more than "\n" -- it also treats U+0085 NEL and U+2028 LINE SEPARATOR
    # (among others) as line boundaries, and git accepts both inside a tag
    # name (confirmed by creating one): a tag such as "v1<NEL>2.0.0" would
    # come back from splitlines() as the two bogus entries "v1" and "2.0.0"
    # instead of the one real tag name git reported. Splitting on a literal
    # "\n" -- the only separator this subprocess's output actually uses --
    # avoids manufacturing tag names that were never in the repository.
    stdout = completed.stdout.decode("utf-8", errors="surrogateescape")
    return [line for line in stdout.split("\n") if line]


def parse_tag_version(tag: str) -> tuple[int, int, int] | None:
    """Parse a ``vMAJOR.MINOR.PATCH`` tag, or return ``None`` if it does not match.

    Tags that do not fit this exact shape (pre-releases, stray annotations)
    are silently excluded from version ordering rather than raising — this
    gate only cares about the newest ``vMAJOR.MINOR.PATCH`` tag in this
    repository, nothing about whether it was published anywhere.

    A component with a leading zero (``v01.0.0``) is also excluded: it is
    not valid semantic versioning, so treating it as a version tag would
    let a malformed spelling parse to the same tuple as the canonical one
    (``v01.0.0`` and ``v1.0.0`` both mean "1") and silently stand in for it
    in :func:`newest_version_tag`'s comparison.

    ``tag`` is matched exactly as given, with no whitespace stripped: a
    name that differs from a canonical tag only by leading or trailing
    whitespace is a different tag and must not be silently treated as the
    one it merely resembles. This matters because ``git check-ref-format``
    only rejects *ASCII* space and control characters in a ref name (verified
    by shelling out to it directly) -- it accepts other Unicode whitespace,
    such as U+00A0 NO-BREAK SPACE, so ``git tag v13.0.0<NBSP>`` succeeds and
    creates a tag distinct from ``v13.0.0``. ``str.strip()`` removes that
    same Unicode whitespace, which previously made the two indistinguishable
    here. A tag our own release tooling creates is always exactly
    ``vMAJOR.MINOR.PATCH`` with no surrounding characters of any kind, so
    nothing legitimate depends on stripping -- only a look-alike tag someone
    else could create would have benefited from it.
    """
    match = TAG_RE.match(tag)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def newest_version_tag(tags: Iterable[str]) -> str:
    """Return the ``v*`` tag with the highest ``MAJOR.MINOR.PATCH`` value.

    Raises :class:`GateInputError` if no tag in ``tags`` matches the
    expected shape.
    """
    candidates = [(parse_tag_version(tag), tag) for tag in tags]
    versioned = [(version, tag) for version, tag in candidates if version is not None]
    if not versioned:
        msg = "no 'vMAJOR.MINOR.PATCH' tag found in this repository"
        raise GateInputError(msg)
    _, newest = max(versioned, key=lambda pair: pair[0])
    return newest


_FULL_OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _ref_path_exists(ref_path: str) -> bool:
    """Return whether ``ref_path`` -- a qualified ref path or a full object ID -- exists.

    The ``^{object}`` suffix forces git to actually resolve an object,
    rather than merely accept the input as syntactically plausible: without
    it, ``git rev-parse --verify`` on a well-formed but nonexistent full
    40-character hex string echoes the string back with exit code 0 and
    never touches the object database, which would make this check a no-op
    for exactly the case it exists to catch.
    """
    completed = subprocess.run(  # noqa: S603 -- trusted internal git invocation
        ["git", "rev-parse", "--verify", "--quiet", f"{ref_path}^{{object}}"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _is_full_object_id(ref: str) -> bool:
    """Return whether ``ref`` is a full 40-character hexadecimal object ID.

    Both hex-digit cases are accepted: git itself resolves an uppercase or
    mixed-case spelling of a full object ID exactly like the lowercase one,
    so refusing uppercase here would reject an input git considers valid.
    """
    return bool(_FULL_OBJECT_ID_RE.match(ref))


def _is_well_formed_ref_path(ref: str) -> bool:
    """Return whether ``ref`` is a ``refs/``-prefixed path git can normalise to a valid ref.

    This is deliberately stricter than "starts with ``refs/``": a revision
    expression such as ``refs/heads/main^0`` or ``refs/heads/main~1`` also
    starts with ``refs/``, but names a ref plus a suffix operation, not the
    ref itself, and must be refused exactly like a bare name is. Delegating
    to ``git check-ref-format --normalize`` catches that -- and every other
    malformed-ref shape -- without this script re-deriving git's own
    ref-name grammar.

    Note that ``--normalize`` means this accepts more than a *literally*
    well-formed path: ``refs//heads///main`` is not itself well-formed, but
    git normalises it to ``refs/heads/main`` and this function returns
    ``True`` for it. That is safe rather than a gap: the caller checks
    whether the *unnormalised* string exists as an object
    (:func:`_ref_path_exists`), and an unnormalised spelling like that one
    does not exist under that exact string, so it is still rejected --
    just later, and with a "does not exist" message rather than a
    "malformed" one.
    """
    if not ref.startswith("refs/"):
        return False
    # No 'text=True': only the exit code is consulted below, but 'text=True'
    # would still make subprocess decode both streams as strict UTF-8 before
    # this function ever saw them. git accepts ref paths containing raw bytes
    # that are not valid UTF-8 (confirmed by creating one in the test suite),
    # '--normalize' echoes the (possibly byte-laden) path back to stdout on
    # success, and neither stream is suppressed the way '--quiet' suppresses
    # it on the other call sites in this module -- so a caller-supplied
    # 'ref' with such bytes previously crashed here with UnicodeDecodeError
    # before reaching the clean exit-1 diagnostic this module promises.
    # Capturing raw bytes and never decoding them removes the crash without
    # needing a decode this function has no use for.
    completed = subprocess.run(  # noqa: S603 -- trusted internal git invocation
        ["git", "check-ref-format", "--normalize", ref],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def assert_ref_is_qualified_and_exists(ref: str) -> None:
    """Raise unless ``ref`` names exactly one thing by construction, and it exists.

    Only two shapes are accepted: a fully qualified ref path beginning with
    ``refs/`` (e.g. ``refs/heads/main``, ``refs/remotes/origin/main``,
    ``refs/tags/v0.6.0``), or a full 40-character hexadecimal object ID.
    Every other shape -- a bare name (``main``), ``HEAD``, an abbreviated
    SHA, or a revision expression (``main^0``) -- is refused outright. Git
    resolves those other shapes by trying a search order over whatever else
    happens to exist in the repository, and that search can silently pick
    something other than what the caller meant; enumerating that search
    order here was tried twice before and both attempts missed real cases,
    which is why this guard no longer tries to reproduce it at all (see the
    module docstring for that history).

    A well-formed ref that does not exist is also rejected here, with a
    message naming it, rather than surfacing as a failure further down the
    call chain.
    """
    if not (_is_full_object_id(ref) or _is_well_formed_ref_path(ref)):
        msg = (
            f"{ref!r} is not an accepted ref. Only a fully qualified ref path "
            f"beginning with 'refs/' (e.g. 'refs/heads/main', "
            f"'refs/remotes/origin/main', 'refs/tags/v0.6.0') or a full "
            f"40-character hexadecimal object ID is accepted -- not a bare "
            f"name, 'HEAD', an abbreviated SHA, or a revision expression such "
            f"as 'main^0', all of which git resolves by searching whatever "
            f"else exists in the repository and can pick something other "
            f"than what you meant. Pass one of the fully qualified forms, or "
            f"a full 40-character commit hash, instead."
        )
        raise GateInputError(msg)
    if not _ref_path_exists(ref):
        msg = f"{ref!r} does not exist in this repository."
        raise GateInputError(msg)


def is_ancestor(ancestor_ref: str, descendant_ref: str) -> bool:
    """Return whether ``ancestor_ref`` is reachable from ``descendant_ref``.

    Raises :class:`GateInputError` if either ref cannot be resolved at all
    (as opposed to resolving but simply not being an ancestor).
    """
    # No 'text=True': git accepts ref names containing raw bytes that are
    # not valid UTF-8 (confirmed by creating one in the test suite), and a
    # descendant_ref built from such a name previously crashed this call
    # with UnicodeDecodeError -- inside subprocess.communicate(), before
    # either branch below ever ran -- instead of the clean exit-1
    # diagnostic this module promises for a bad input.
    completed = subprocess.run(  # noqa: S603 -- trusted internal git invocation
        ["git", "merge-base", "--is-ancestor", ancestor_ref, descendant_ref],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode in (0, 1):
        return completed.returncode == 0
    # Decoded with 'surrogateescape', the same convention list_git_tags()
    # uses and for the same reason: it preserves every byte losslessly
    # instead of raising here. The result can still contain an unpaired
    # surrogate that cannot be encoded back to UTF-8 -- that is handled once,
    # centrally, at the point this message is actually written (see
    # main()), not here.
    stderr = completed.stderr.decode("utf-8", errors="surrogateescape")
    msg = (
        f"'git merge-base --is-ancestor {ancestor_ref} {descendant_ref}' "
        f"could not resolve one of the refs: {stderr.strip()}"
    )
    raise GateInputError(msg)


def _resolves_to_a_local_branch(ref: str) -> bool:
    """Return whether ``ref`` ultimately names a local branch.

    Only a local branch can be the target of a forward merge -- forwarding
    ``dev`` into a tag, a remote-tracking ref, or a commit is not an action
    a caller can take, so this decides whether that remediation may be
    offered. By the time this runs, ``ref`` has already passed
    :func:`assert_ref_is_qualified_and_exists`, so it is always either a
    full object ID or a fully qualified ref path -- an object ID never
    starts with ``refs/heads/`` (SHAs are not stored as ref names by
    convention), so any object ID is handled by the prefix check below
    returning ``False`` without needing to shell out.

    A ``refs/heads/*`` prefix alone is not sufficient, though: a ref stored
    under ``refs/heads/`` can itself be a *symbolic* ref pointing somewhere
    else entirely (``git symbolic-ref refs/heads/alias refs/tags/v1.0.0`` is
    accepted by git and later resolves through to the tag). A plain prefix
    check would call that a local branch and prescribe a forward merge that
    is not actually possible. This resolves the symbolic link, if there is
    one, and judges the prefix on the resolved target instead.

    No ``text=True``: git accepts a symbolic ref whose *target* contains raw
    bytes that are not valid UTF-8 (confirmed by creating one in the test
    suite) -- a branch name is not required to be UTF-8, only well-formed by
    ``git check-ref-format``'s rules, and a target built from such a name
    reaches here unchanged. ``git symbolic-ref --quiet`` succeeds and echoes
    that target on stdout; ``text=True`` decodes it eagerly as strict UTF-8
    before this function ever inspects it, raising ``UnicodeDecodeError``
    inside ``subprocess.communicate()`` -- outside :class:`GateInputError`,
    so it would escape ``main()`` as a bare traceback instead of the clean
    exit-1 diagnostic this module promises. Comparing the raw prefix bytes
    directly avoids the decode entirely: this function only ever needs to
    know whether the target starts with ``refs/heads/``, a question raw
    bytes answer exactly as well as text does.
    """
    if not ref.startswith("refs/heads/"):
        return False
    completed = subprocess.run(  # noqa: S603 -- trusted internal git invocation
        ["git", "symbolic-ref", "--quiet", ref],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        # Not a symbolic ref -- an ordinary ref under refs/heads/ names a
        # local branch by definition.
        return True
    return completed.stdout.strip().startswith(b"refs/heads/")


def run_gate(*, branch: str) -> tuple[bool, list[str]]:
    """Check whether the newest version tag is an ancestor of ``branch``.

    Returns ``(passed, report_lines)``.
    """
    assert_ref_is_qualified_and_exists(branch)
    tags = list_git_tags()
    tag = newest_version_tag(tags)
    # ``tag`` is a bare name from 'git tag -l' (e.g. "v12.0.0"), not a
    # qualified ref path. Passing it to is_ancestor() as-is would let git
    # resolve it through its usual bare-name search order -- exactly the
    # ambiguity assert_ref_is_qualified_and_exists() above exists to close
    # for 'branch', but that guard was never applied to the tag side. A
    # same-named ref elsewhere in the search order (e.g. a top-level
    # 'refs/<tag>' or a branch) would then be resolved instead of the real
    # tag, silently. Qualifying it as 'refs/tags/<tag>' removes that
    # ambiguity the same way 'branch' is already required to be qualified.
    # The bare 'tag' is still what gets printed to the reader below.
    contains_tag = is_ancestor(f"refs/tags/{tag}", branch)

    messages = [
        f"newest version tag: {tag}",
        f"branch checked:      {branch}",
    ]
    if contains_tag:
        messages.append(f"PASS: {tag!r} is reachable from {branch!r}.")
    elif _resolves_to_a_local_branch(branch):
        messages.append(
            f"FAIL: {tag!r} is not reachable from {branch!r} -- forward-merge 'dev' "
            f"into {branch!r} after confirming {tag!r} published successfully to PyPI.",
        )
    else:
        messages.append(f"FAIL: {tag!r} is not reachable from {branch!r}.")
    return contains_tag, messages


def _safe_for_stream(text: str, stream: TextIO) -> str:
    r"""Return ``text`` re-encoded so writing it to ``stream`` cannot raise ``UnicodeEncodeError``.

    ``text`` can carry an unpaired surrogate -- either from git's own error
    text, decoded with ``errors="surrogateescape"`` in :func:`list_git_tags`
    and :func:`is_ancestor` for the same reason those functions use that
    codec (see their docstrings), or directly from a caller-supplied
    ``--branch`` value that reached Python as one in the first place,
    because the OS could not decode *it* as UTF-8 either (this is how
    CPython represents an undecodable ``argv`` entry; see PEP 383) and it
    was echoed back into a message or report line unchanged. A surrogate is
    not the only way a write can fail, though: ``text`` can just as easily
    be a perfectly ordinary, valid Unicode ref such as ``refs/heads/é`` --
    valid UTF-8, and encodable as such -- that still fails to encode in
    whatever ``stream`` is actually configured with, if that is stricter
    than UTF-8 (ASCII, for instance).

    An earlier version of this function hardcoded ``"utf-8"`` rather than
    consulting ``stream``, so its only real guarantee was "UTF-8 encodable"
    -- which a strict ASCII ``sys.stdout`` (or any stream whose encoding is
    not UTF-8) can still reject even after that sanitising, as confirmed by
    execution: ``refs/heads/é`` reached a strict-ASCII stream unchanged and
    raised ``UnicodeEncodeError`` there, despite this function's docstring
    at the time claiming it made "a write to sys.stderr" safe outright.
    Reading ``stream.encoding`` and re-encoding *against that* instead makes
    the guarantee real rather than merely narrowing what it claims: whatever
    ``stream`` turns out to be configured with, the text handed to
    ``stream.write`` is already representable in that same encoding, so the
    write itself cannot raise on encoding grounds. A stream that reports no
    ``encoding`` attribute (not a real ``TextIOWrapper``) falls back to
    ``"utf-8"``, the previous behaviour, rather than failing outright.

    That guarantee assumes ``stream.encoding``, when set, names a codec
    Python's codec registry actually has -- true of every real
    ``io.TextIOWrapper`` (including ``sys.stdout``/``sys.stderr`` as CPython
    constructs them), because ``TextIOWrapper.__init__`` performs the same
    codec lookup itself and raises ``LookupError`` immediately, before the
    stream can exist at all, if the name is not registered (confirmed by
    execution: ``io.TextIOWrapper(io.BytesIO(), encoding="x-no-such-codec")``
    raises ``LookupError`` at construction, never at write time). This
    function does *not* cover the case this rules out for every real stream:
    a duck-typed object whose ``.encoding`` attribute names something the
    codec registry does not recognise reaches ``text.encode(encoding, ...)``
    below and raises ``LookupError`` there instead of returning safely --
    confirmed by execution against exactly such a fake. That is a narrower
    promise than "safe for any object with an ``.encoding`` attribute", and
    deliberately so: guarding against a codec name no real ``TextIOWrapper``
    can ever report would mean choosing a silent fallback for a fabricated
    input, which is how the previous, hardcoded-``"utf-8"`` version of this
    function came to overstate its own guarantee in the first place.

    ``"backslashreplace"`` is used for both the encode and the decode: on
    encode, it turns each character ``stream``'s encoding cannot represent
    (an unpaired surrogate, or a code point outside a narrower charset like
    ASCII) into a literal ``\\xHH``/``\\uHHHH``-style escape made of plain
    ASCII bytes -- always representable in any of these encodings, so the
    decode back to ``str`` cannot fail either. The cost is fidelity, not
    safety: the exact original character is no longer reproduced, only its
    escaped spelling -- acceptable here because this string exists to be
    read by a human deciding whether a ``--branch`` value was well-formed,
    not to be parsed back into the original text.
    """
    encoding = getattr(stream, "encoding", None) or "utf-8"
    raw = text.encode(encoding, errors="backslashreplace")
    return raw.decode(encoding, errors="backslashreplace")


def _write(stream: TextIO, text: str) -> None:
    """Write ``text`` plus a trailing newline to ``stream`` through the one sanitising choke point.

    Every write ``main()`` makes -- on both the success path and the error
    path -- goes through this single function rather than calling
    ``stream.write`` directly. A previous round only wrapped the one
    interpolation inside the ``except GateInputError`` branch; that covered
    the message built from a caught error, but ``run_gate()``'s
    success-path report also interpolates the checked ``branch`` verbatim
    into a "branch checked: ..." line, and that write was never routed
    through any sanitiser -- it went straight to ``sys.stdout.write``. An
    *existing* branch whose name is not valid UTF-8 passes both validation
    and the ancestry check without incident and then crashes while the
    result is being printed.

    Sanitising once here, at the point every write funnels through, was
    chosen over sanitising each interpolation site individually (the
    approach the previous round took) because a per-site fix only covers
    the sites someone remembered to wrap -- which is exactly how the
    success-path write was missed the first time. A single choke point
    means a write added later is safe by construction, without depending on
    its author to know which values can carry an unpaired surrogate, or
    what encoding the stream it lands on actually has (see
    :func:`_safe_for_stream`).
    """
    stream.write(_safe_for_stream(text, stream) + "\n")


class _SanitizingArgumentParser(argparse.ArgumentParser):
    r"""``ArgumentParser`` whose help/usage/error writes go through this module's one choke point.

    ``ArgumentParser.parse_args()`` can write directly to ``sys.stdout``
    (``--help``) or ``sys.stderr`` (an unrecognised option, a missing
    argument value) without ever calling :func:`_write` -- confirmed by
    execution: instrumenting ``_write`` and calling ``main(["--help"])`` or
    ``main(["--bogus"])`` against the base ``argparse.ArgumentParser``
    recorded zero calls to it. ``_write``'s own docstring promises that
    *every* write ``main()`` makes goes through this one sanitising choke
    point; before this subclass, argparse's own writes were the gap that
    made that promise false. The practical cost is real, not merely
    inconsistent: with a strict-UTF-8 ``sys.stderr`` installed,
    ``main(["bad-\\udcff"])`` used to raise ``UnicodeEncodeError`` -- from
    argparse's own unsanitised ``file.write(message)`` -- after printing
    only the usage line, instead of the clean exit-2 diagnostic naming the
    bad argument.

    Both writing paths already funnel through one base-class method,
    ``_print_message(message, file)`` -- a bare ``file.write(message)`` with
    no sanitising at all -- so overriding only that one method is enough to
    cover both: ``print_help()`` (``--help``) calls it with
    ``file=sys.stdout``, and ``error()`` calls it twice, once via
    ``print_usage(sys.stderr)`` and once via ``exit(2, message)``, both with
    ``file=sys.stderr``.
    """

    def _print_message(self, message: str, file: SupportsWrite[str] | None = None) -> None:
        if not message:
            return
        # The base class types 'file' as the minimal 'SupportsWrite[str]',
        # but argparse itself only ever calls this with sys.stdout or
        # sys.stderr (see this class's docstring) -- both real TextIO
        # objects, which is what _write() (and _safe_for_stream() inside
        # it) needs to read '.encoding' off of.
        stream = cast("TextIO", file) if file is not None else sys.stdout
        # argparse's own messages already end with their own trailing
        # newline; _write() appends one more, so strip exactly one here to
        # avoid a doubled blank line relative to the pre-fix output.
        _write(stream, message.removesuffix("\n"))


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the main-carries-the-released-tag check."""
    parser = _SanitizingArgumentParser(
        prog="check_main_carries_the_released_tag.py",
        description=(
            "Fail when the newest 'vMAJOR.MINOR.PATCH' tag in this repository "
            "is not an ancestor of the given branch ('refs/heads/main' by default)."
        ),
    )
    parser.add_argument(
        "--branch",
        default="refs/heads/main",
        help=(
            "fully qualified ref (e.g. 'refs/heads/main', 'refs/remotes/origin/main', "
            "'refs/tags/v0.6.0') or full 40-character commit SHA to check for the "
            "newest version tag (default: 'refs/heads/main')"
        ),
    )
    args = parser.parse_args(argv)

    _write(sys.stdout, "=" * 60)
    _write(sys.stdout, "Main-carries-the-released-tag check")
    _write(sys.stdout, "=" * 60)

    try:
        passed, messages = run_gate(branch=args.branch)
    except GateInputError as exc:
        _write(sys.stderr, f"main-carries-the-released-tag check: {exc}")
        return EXIT_FAIL

    for line in messages:
        _write(sys.stdout, line)
    _write(sys.stdout, "=" * 60)
    verdict = "PASS" if passed else "FAIL"
    _write(sys.stdout, f"MAIN CARRIES THE VERSION TAG: {verdict}")
    _write(sys.stdout, "=" * 60)

    return EXIT_OK if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
