"""Layer 7 of the documentation-example tests — ``yaml`` config vs the real loader.

Every existing documentation-test layer asks the Markdown extractor for a
``python`` fenced block (one, ``test_docs_examples_behavior.py``'s tutorial
and upgrade checks, also compares ``text`` transcripts against real output).
Nothing has ever looked at ``yaml``, so a documented ``engrava.yaml`` key that
no configuration class backs left the whole suite green — and it is the
sharpest form of this bug, because ``load_config`` already raises
``ConfigError: Unknown configuration key(s)`` for exactly this case: a reader
who pastes the documented snippet gets an immediate failure.

Delegating to the real loader, not reimplementing it
------------------------------------------------------
Two earlier versions of this module were both wrong in review-caught ways.
The first mirrored each section's key set by hand and split a block into
"stanzas"; the mirror missed real value-level checks and the splitter broke
comments, ``null``, and anchors/aliases. The second delegated to
``load_config`` but *preprocessed the text* before parsing it: substituting
``${VAR}`` placeholders before ``yaml.safe_load`` let substitution repair
genuinely invalid YAML (a flow mapping broken by an unescaped placeholder),
and unconditionally injecting a placeholder ``database:`` section let a
documented **complete** ``engrava.yaml`` example lose a real defect (a
missing ``database.path``) to the very placeholder meant only for fragments.

This version fixes both by keeping the raw document intact until it is a
real Python value, and by only ever inventing content for a block the
registry below says needs it:

0. **Check for a duplicate written key on the composed node graph, before
   anything is constructed.** Three earlier versions of this one check got
   the ordering wrong in three different directions: a plain
   ``yaml.safe_load`` silently keeps only the last of a duplicate key; a
   ``SafeLoader.construct_mapping`` override that checked *before*
   flattening a merge key crashed on the merge key's own node (which has no
   registered constructor outside ``flatten_mapping``'s own handling of it);
   checking *after* flattening mistook an explicit key overriding a merged
   one — or two merge sources that legitimately overlap — for a real
   duplicate, and a shared node reached through more than one alias made
   construction order matter in a way it never should have. "Is this key
   written twice in this mapping" is a property of the document's *text*,
   not of a constructed value, so ``yaml.compose()`` gets the node graph
   (composed, not constructed — no construction and no merge flattening;
   ``yaml.compose()`` does still resolve an alias to the *same* mapping
   node object its anchor produced, which is exactly what lets a duplicate
   inside a shared, aliased mapping still be caught) and
   ``_find_duplicate_key`` walks it directly: for each ``MappingNode``, its
   own written keys, skipping the merge key itself. See that function's
   docstring for the full history.
1. **Parse first.** ``yaml.safe_load`` runs on the block exactly as written.
   Invalid YAML fails as invalid YAML — nothing before this step touches the
   text.
2. **Substitute after parsing, and only at ``embeddings.api_key``.** That
   field alone is a string shaped like ``${VAR}`` replaced with a fixed
   dummy — never in the source text, and never anywhere else in the
   document. It is the one field the real loader itself interpolates
   (``_resolve_env_var``, called only from ``_parse_embeddings``, for a
   top-level or a per-service ``embeddings:`` section), and it raises if the
   referenced variable is unset — so leaving it unresolved would make this
   checker's verdict depend on who is running it. A placeholder anywhere
   else (``services.default_service``, for one) is handed to the loader
   exactly as written, because the loader does not interpolate it either;
   substituting it there would validate a document a real reader pasting
   the same text would not get. See ``_substitute_env_placeholders``.
3. **Classify each block as complete or a fragment**, the same way exempt
   blocks are registered (see ``COMPLETE_YAML_BLOCKS`` below). A **complete**
   block — one the docs present as the entire ``engrava.yaml`` a reader would
   create — goes to ``load_config`` untouched: no injected value, so a defect
   like a missing ``database.path`` is caught, not repaired. A **fragment**
   — a single section shown in isolation, which is what almost every
   documented block actually is — gets a placeholder ``database.path``
   injected (the one field ``load_config`` unconditionally requires that this
   checker is not testing), so it can be checked at all. For a fragment, the
   check is key existence and value validity **within what is shown**, not
   completeness — a fragment is never proof the reader's whole file works.
4. **An empty document and an explicit ``null`` are not the same.** A block
   with no content at all has nothing to check. A block whose content is the
   literal ``null`` (or a bare ``~``, or only comments) is a value the real
   reader's parser would also produce ``None`` for, and ``load_config``
   rejects a non-mapping document — so it is checked, and correctly fails.
5. Write the (possibly substituted, possibly database-augmented) document to
   a temporary file and call ``load_config`` on it, catching ``ConfigError``.

Any ``ConfigError`` the real loader raises is reported, and only that: there
is no mirrored key or value model left in this module to drift from the
shipped classes.

Fragment vs. complete, honestly
--------------------------------
Only **3 of the 25** documented ``yaml`` blocks already include a
``database:`` section and are registered as complete
(``docs/configuration.md``'s "Create a ``engrava.yaml`` file:" example,
and two others that happen to show it alongside one more section). The
other 20 checkable blocks are fragments. This is not a defect to fix by
reclassifying more blocks as complete — a snippet titled "Configure it via
``SearchConfig``:" or "A minimal enable:" is genuinely illustrating one
section, not publishing a runnable file — but it does mean the
"key exists and the value is valid" guarantee is real for every block, while
the *stronger* "this whole example runs as-is" guarantee this module could
offer only applies to those 3.

Two documented blocks each show the same top-level key (``manifests:``)
twice, side by side, to illustrate alternate forms. That is a real duplicate
key by construction, so the duplicate-key check (correctly) reports it —
checking either whole block as one document can only ever report the
duplicate, never validate either form on its own. Both blocks are
registered as exempt (``ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS``)
rather than silently under-checked.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from engrava import ConfigError, load_config
from tests.docs._md_blocks import (
    CodeBlock,
    ExemptionReason,
    extract_exact_fenced_blocks,
    markdown_files,
)


def _all_yaml_blocks() -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    for path in markdown_files():
        blocks.extend(extract_exact_fenced_blocks(path, "yaml"))
    return blocks


_ALL_YAML_BLOCKS = _all_yaml_blocks()

# YAML blocks this module cannot check, each with a reason from the closed
# ExemptionReason vocabulary. The two entries today each show the same
# top-level key twice to illustrate alternate forms -- a real duplicate key
# by construction, so the duplicate-key check reports it, and checking the
# whole block as one document can never validate either form on its own.
EXEMPT_YAML_BLOCKS: tuple[tuple[str, str, ExemptionReason], ...] = (
    (
        "docs/configuration.md",
        "# list form",
        ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS,
    ),
    (
        "docs/extensions.md",
        "# Explicit dotted paths",
        ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS,
    ),
)

# YAML blocks that already document a complete engrava.yaml (they include a
# `database:` section themselves) and so must go to load_config exactly as
# written, with no injected placeholder -- deleting part of one of these is a
# real documentation defect (e.g. `'database.path' is required`), not
# something an injected value should paper over. Every checked block NOT
# listed here is treated as a fragment (see the module docstring for what
# that does and does not guarantee).
COMPLETE_YAML_BLOCKS: tuple[tuple[str, str], ...] = (
    ("docs/configuration.md", "wal_mode: true"),
    ("docs/audit-trail.md", '"./engrava.db"'),
    ("docs/backup-and-recovery.md", "path: ./fresh.db"),
)

# A shell-style placeholder a documented example may show for a value a reader
# is expected to supply, e.g. `${OPENAI_API_KEY}`. `engrava.config` itself
# resolves exactly this syntax, but only at `embeddings.api_key` -- the sole
# call site of its own `_resolve_env_var`, which supports only a single
# `${...}` wrapping the *entire* value, no inline interpolation -- and
# raises `ConfigError` if the named variable is unset. Leaving that one
# field's placeholder unresolved would make this checker's verdict depend on
# the *test runner's* environment, not on anything the example claims, so it
# is substituted (in the parsed structure, never the source text -- see the
# module docstring). Every other `${...}` anywhere else in a document is
# handed to the loader untouched: the real loader never interpolates it
# either, so substituting it would validate something a reader would not
# get -- exactly the class of bug (preprocessing making something valid that
# is not) this module exists to avoid.
_ENV_PLACEHOLDER_DUMMY = "doc-check-dummy-value"

# The one field `load_config` unconditionally requires (`database.path`) even
# though this checker is not testing it. Injected only for a block registered
# as a fragment above, and only when the document has no `database:` section
# of its own.
_PLACEHOLDER_DATABASE_PATH = "doc-example.db"


_MERGE_TAG = "tag:yaml.org,2002:merge"


def _find_duplicate_key(node: yaml.Node, visited: set[int]) -> str | None:
    """Walk a composed (unconstructed) node graph for a mapping with a repeated key.

    "Is this key written twice in this mapping" is a property of the
    document's text, not of the value construction gets to -- so this
    inspects ``yaml.compose()``'s node graph directly, before anything is
    constructed. That sidesteps every ordering bug a constructor override
    had: ``flatten_mapping`` folding a merge key's referenced keys in before
    an override sees them (raising on the merge key's own, unconstructable
    node), or after (mistaking an explicit key overriding a merged one, or
    two merge sources that legitimately overlap, for a real duplicate), and
    an alias being resolved into a *shared* node object that construction
    order could see mutated by an earlier reference to it.

    The YAML merge key (``<<:``) is not a key at all -- it is a directive,
    identified by its own node's special ``tag:yaml.org,2002:merge`` tag,
    and is skipped here rather than compared as one. A duplicate *inside* a
    merged or anchored mapping is still caught: it is a distinct
    ``MappingNode`` object, visited (and checked) on its own regardless of
    how many places reference it -- ``visited`` exists only to check that
    object once, not to widen or narrow what counts as a duplicate.

    Returns:
        The first repeated key's text, or ``None`` if none is found.

    """
    if id(node) in visited:
        return None
    visited.add(id(node))

    children: list[yaml.Node] = []
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.tag != _MERGE_TAG:
                key_text = str(key_node.value)
                if key_text in seen:
                    return key_text
                seen.add(key_text)
            children.append(key_node)
            children.append(value_node)
    elif isinstance(node, yaml.SequenceNode):
        children.extend(node.value)
    # A ScalarNode leaves `children` empty; nothing to recurse into.

    for child in children:
        found = _find_duplicate_key(child, visited)
        if found is not None:
            return found
    return None


def _is_whole_value_placeholder(value: object) -> bool:
    """Whether ``value`` is a string that is a single ``${VAR}`` wrapping its whole length.

    Mirrors ``_resolve_env_var``'s own check exactly (``startswith("${")``
    and ``endswith("}")``) -- that function supports no inline
    interpolation, so a placeholder embedded in a longer string (or one
    that is not the field's entire value) is not something the real loader
    resolves either, and must not be substituted here.
    """
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def _substitute_api_key_placeholder(embeddings: object) -> object:
    """Substitute a whole-value ``${VAR}`` placeholder at one ``embeddings.api_key``.

    Returns ``embeddings`` unchanged unless it is a mapping with an
    ``api_key`` that is exactly one placeholder end to end.
    """
    if not isinstance(embeddings, dict) or "api_key" not in embeddings:
        return embeddings
    if not _is_whole_value_placeholder(embeddings["api_key"]):
        return embeddings
    substituted = dict(embeddings)
    substituted["api_key"] = _ENV_PLACEHOLDER_DUMMY
    return substituted


def _substitute_env_placeholders(parsed: Any) -> Any:  # noqa: ANN401
    """Substitute a placeholder only at ``embeddings.api_key``, top-level or per-service.

    ``embeddings.api_key`` is the *only* field the real loader itself
    interpolates (``_resolve_env_var``, called from exactly one site,
    ``_parse_embeddings``) -- and that function is called both for a
    top-level ``embeddings:`` section and for each
    ``services.configs.<name>.embeddings:`` section, so both are covered.
    Every other ``${...}`` anywhere in the document, at any key, is left
    exactly as written and handed to the loader untouched: the loader does
    not interpolate it either, so substituting it would validate a document
    a real reader pasting the same text would not get. Operates on the
    value *after* ``yaml.safe_load``, never on source text -- substituting
    in the text could repair genuinely invalid YAML (e.g. a flow mapping
    broken by an unescaped ``${...}``), which must fail as invalid YAML
    instead.
    """
    if not isinstance(parsed, dict):
        return parsed
    result = dict(parsed)
    if "embeddings" in result:
        result["embeddings"] = _substitute_api_key_placeholder(result["embeddings"])
    services = result.get("services")
    if isinstance(services, dict) and isinstance(services.get("configs"), dict):
        services = dict(services)
        configs = dict(services["configs"])
        for name, raw_service in configs.items():
            if isinstance(raw_service, dict) and "embeddings" in raw_service:
                updated_service = dict(raw_service)
                updated_service["embeddings"] = _substitute_api_key_placeholder(
                    raw_service["embeddings"],
                )
                configs[name] = updated_service
        services["configs"] = configs
        result["services"] = services
    return result


def _inject_placeholder_database(parsed: Any) -> Any:  # noqa: ANN401
    """Add a placeholder ``database.path`` to a fragment that needs one.

    A fragment may omit ``database`` entirely (most do), or may show
    ``database`` with some other key (e.g. ``wal_mode``) but no ``path`` --
    both are judged on what they show, so both get exactly the missing
    ``path`` value injected, never replacing a section the fragment already
    documents. A non-mapping document, or one whose own ``database`` is not
    a mapping, is passed through untouched so ``load_config`` reports its
    own, real error for it.
    """
    if not isinstance(parsed, dict):
        return parsed
    if "database" not in parsed:
        return {"database": {"path": _PLACEHOLDER_DATABASE_PATH}, **parsed}
    database_section = parsed["database"]
    if isinstance(database_section, dict) and "path" not in database_section:
        augmented = dict(parsed)
        augmented["database"] = {"path": _PLACEHOLDER_DATABASE_PATH, **database_section}
        return augmented
    return parsed


@dataclass(frozen=True)
class ConfigExampleError:
    """The single ``ConfigError`` the real loader raised for a documented block."""

    message: str


def block_config_error(body: str, *, complete: bool) -> ConfigExampleError | None:
    """Validate one documented YAML block via the real, unmodified ``load_config``.

    Args:
        body: The fenced block's text, exactly as documented.
        complete: Whether the block is registered as a complete
            ``engrava.yaml`` example (see ``COMPLETE_YAML_BLOCKS``). A
            complete block is never augmented; a fragment gets a placeholder
            ``database.path`` when it has none of its own.

    Returns:
        A :class:`ConfigExampleError` for the first problem found, or
        ``None`` if none is. A duplicate written key (see
        ``_find_duplicate_key``) is checked, and reported, before
        ``load_config`` ever runs; past that point ``load_config`` itself
        fails fast on the first problem it meets, so a document with more
        than one defect reports only the first. A block with no content at
        all (as opposed to one whose content is literally ``null``) is not
        an error: there is nothing to check.

    """
    if not body.strip():
        return None  # No content at all -- distinct from an explicit `null`.
    try:
        root = yaml.compose(body)
    except yaml.YAMLError as exc:
        return ConfigExampleError(f"invalid YAML: {exc}")
    if root is not None:
        duplicate = _find_duplicate_key(root, set())
        if duplicate is not None:
            return ConfigExampleError(f"duplicate key {duplicate!r} in mapping")

    parsed = yaml.safe_load(body)
    parsed = _substitute_env_placeholders(parsed)
    if not complete:
        parsed = _inject_placeholder_database(parsed)

    fd, temp_path_str = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(parsed, handle)
        load_config(temp_path_str)
    except ConfigError as exc:
        return ConfigExampleError(str(exc))
    finally:
        Path(temp_path_str).unlink(missing_ok=True)
    return None


def _unique_block(rel: str, anchor: str) -> CodeBlock:
    matches = [b for b in _ALL_YAML_BLOCKS if b.rel == rel and anchor in b.body]
    if len(matches) != 1:
        pytest.fail(
            f"anchor {anchor!r} matched {len(matches)} yaml blocks in {rel} (want "
            f"exactly 1); update the registry in {__file__}.",
        )
    return matches[0]


def _exempt_locations() -> dict[str, ExemptionReason]:
    return {
        _unique_block(rel, anchor).location: reason for rel, anchor, reason in EXEMPT_YAML_BLOCKS
    }


def _complete_locations() -> set[str]:
    return {_unique_block(rel, anchor).location for rel, anchor in COMPLETE_YAML_BLOCKS}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _block_id(block: CodeBlock) -> str:
    return block.location


def test_yaml_extractor_found_blocks() -> None:
    """Sanity check against a silently-empty extractor."""
    assert len(_ALL_YAML_BLOCKS) > 10, (
        f"expected many yaml blocks, found {len(_ALL_YAML_BLOCKS)}; the "
        f"extractor may be misconfigured."
    )


def test_exempt_yaml_registry_anchors_are_unique() -> None:
    """Every EXEMPT_YAML_BLOCKS anchor binds exactly one yaml block."""
    _exempt_locations()


def test_complete_yaml_registry_anchors_are_unique() -> None:
    """Every COMPLETE_YAML_BLOCKS anchor binds exactly one yaml block."""
    locations = _complete_locations()
    assert len(locations) == len(COMPLETE_YAML_BLOCKS)


def test_every_yaml_block_is_covered() -> None:
    """Every yaml block is either checked (zero real-loader errors) or exempt.

    Exemption here means "cannot be checked at all" (its whole document, as
    written, cannot be handed to ``load_config`` and prove anything about
    every form it shows), not "has no violations" — a checked block with a
    violation still fails in
    ``test_documented_config_keys_exist_on_the_real_classes`` below, not here.
    """
    exempt = _exempt_locations()
    complete = _complete_locations()
    checked = [b for b in _ALL_YAML_BLOCKS if b.location not in exempt]
    assert len(checked) + len(exempt) == len(_ALL_YAML_BLOCKS)
    print(  # noqa: T201 — intentional census summary for the -s report
        f"\nYAML-example census: total={len(_ALL_YAML_BLOCKS)} "
        f"checked={len(checked)} (complete={len(complete)} "
        f"fragment={len(checked) - len(complete)}) exempt={len(exempt)}"
    )
    reason_tally: dict[str, int] = {}
    for reason in exempt.values():
        reason_tally[reason.value] = reason_tally.get(reason.value, 0) + 1
    for reason_value, count in sorted(reason_tally.items()):
        print(f"  exempt[{reason_value}] = {count}")  # noqa: T201


@pytest.mark.parametrize("block", _ALL_YAML_BLOCKS, ids=[_block_id(b) for b in _ALL_YAML_BLOCKS])
def test_documented_config_keys_exist_on_the_real_classes(block: CodeBlock) -> None:
    """The real, unmodified ``load_config`` accepts every documented YAML block.

    A key no dataclass field backs, or a value the real loader rejects (a
    non-boolean ``wal_mode``, a negative ``dimension``, ...), is an example
    that would fail for the reader who pastes it. A block registered as
    complete is checked exactly as written; a fragment is checked with a
    placeholder ``database.path`` (see the module docstring for what that
    does and does not prove).
    """
    exempt = _exempt_locations()
    if block.location in exempt:
        pytest.skip("exempt yaml block: cannot be checked against engrava.config")
    complete = block.location in _complete_locations()
    error = block_config_error(block.body, complete=complete)
    assert error is None, (
        f"Documentation yaml block at {block.location} is rejected by the real "
        f"load_config: {error.message}"
    )


# ---------------------------------------------------------------------------
# Failability + controls (direct unit checks of the checker logic itself,
# using the exact mutation text from the finding)
# ---------------------------------------------------------------------------


def test_checker_rejects_an_unknown_nested_key() -> None:
    """The exact mutation: `search.hybrid_fusion_mode` is not a real field."""
    error = block_config_error("search:\n  hybrid_fusion_mode: reciprocal_rank\n", complete=False)
    assert error is not None
    assert "hybrid_fusion_mode" in error.message


def test_checker_rejects_an_unknown_top_level_key() -> None:
    """The exact mutation: `turbo` is not a real top-level section."""
    error = block_config_error("turbo:\n  enabled: true\n", complete=False)
    assert error is not None
    assert "turbo" in error.message


def test_checker_rejects_the_full_verbatim_mutation() -> None:
    """The finding's exact appended snippet is rejected by the real loader.

    `load_config` fails fast: the top-level key check runs before any
    section's own parser, so `turbo` (an unknown top-level key) is what
    surfaces here, not `hybrid_fusion_mode` -- both are independently proven
    broken by the two tests above.
    """
    mutation = "search:\n  hybrid_fusion_mode: reciprocal_rank\nturbo:\n  enabled: true\n"
    error = block_config_error(mutation, complete=False)
    assert error is not None
    assert "turbo" in error.message


def test_checker_accepts_a_valid_deeply_nested_key_on_a_non_obvious_class() -> None:
    """Control: a real, valid, deeply-nested key must not fire.

    `DreamingGates.cluster_quality_persona_threshold` lives three levels deep
    (`extensions.dreaming.gates.*`) on a class with 17 fields, about as
    "non-obvious" as this schema gets.
    """
    fragment = (
        "extensions:\n  dreaming:\n    gates:\n      cluster_quality_persona_threshold: 0.8\n"
    )
    assert block_config_error(fragment, complete=False) is None


def test_checker_accepts_a_null_hooks_section() -> None:
    """Regression: `hooks: null` is accepted by the real loader (means "unset").

    A mirrored validator that requires `hooks` to be a mapping whenever the
    key is present would reject this; the real loader does not.
    """
    assert block_config_error("hooks: null\n", complete=False) is None


def test_checker_does_not_let_a_column_zero_comment_split_a_mapping() -> None:
    """Regression: a comment between two keys of one mapping is not a new document."""
    fragment = (
        "search:\n"
        "  default_fts_weight: 0.30\n"
        "\n"
        "# More search settings\n"
        "  default_vector_weight: 0.55\n"
    )
    assert block_config_error(fragment, complete=False) is None


def test_checker_accepts_an_anchor_and_alias_spanning_two_sections() -> None:
    """Regression: a YAML anchor defined in one section and used in another works."""
    fragment = "database:\n  path: &db_path demo.db\n\nservices:\n  data_dir: *db_path\n"
    assert block_config_error(fragment, complete=True) is None


def test_checker_rejects_a_non_boolean_wal_mode() -> None:
    """Regression: the real loader rejects a non-boolean `wal_mode`; a mirror missed this."""
    error = block_config_error(
        'database:\n  path: demo.db\n  wal_mode: "yes"\n',
        complete=True,
    )
    assert error is not None
    assert "wal_mode" in error.message


def test_checker_rejects_a_negative_vector_dimension() -> None:
    """Regression: the real loader rejects a negative dimension; a mirror missed this."""
    error = block_config_error("extensions:\n  vector:\n    dimension: -1\n", complete=False)
    assert error is not None


def test_checker_substitutes_an_env_placeholder_deterministically() -> None:
    """A placeholder outside `embeddings.api_key` is left as literal text, untouched.

    `hooks.class` is a dotted-import-path string with no content validation
    beyond being a string, so the literal `${MY_SECRET}` text passes here --
    not because it was substituted (it was not; only `embeddings.api_key`
    is), but because nothing about this field depends on the environment in
    the first place.
    """
    assert block_config_error('hooks:\n  class: "${MY_SECRET}"\n', complete=False) is None


def test_checker_rejects_a_placeholder_outside_embeddings_api_key() -> None:
    """Regression: substitution must not repair a field the real loader never interpolates.

    `services.default_service` is validated as a literal service name; the
    real loader raises `Invalid service name '${DOC_SERVICE}'` even with
    `DOC_SERVICE` set, because `_resolve_env_var` is never called for this
    field. Substituting a dummy here would validate a document a real
    reader pasting the same text would not get.
    """
    fragment = 'services:\n  data_dir: ./data\n  default_service: "${DOC_SERVICE}"\n'

    os.environ.pop("DOC_SERVICE", None)
    error_without = block_config_error(fragment, complete=False)
    assert error_without is not None
    assert "Invalid service name" in error_without.message
    assert "${DOC_SERVICE}" in error_without.message

    os.environ["DOC_SERVICE"] = "main"
    try:
        error_with = block_config_error(fragment, complete=False)
    finally:
        del os.environ["DOC_SERVICE"]
    assert error_with is not None
    assert "Invalid service name" in error_with.message


def test_checker_accepts_an_embeddings_api_key_placeholder_regardless_of_environment() -> None:
    """Control: the one field the real loader does interpolate stays deterministic."""
    fragment = 'embeddings:\n  provider: openai-compatible\n  api_key: "${OPENAI_API_KEY}"\n'

    os.environ.pop("OPENAI_API_KEY", None)
    assert block_config_error(fragment, complete=False) is None

    os.environ["OPENAI_API_KEY"] = "sk-whatever"
    try:
        assert block_config_error(fragment, complete=False) is None
    finally:
        del os.environ["OPENAI_API_KEY"]


def test_checker_accepts_a_per_service_embeddings_api_key_placeholder() -> None:
    """Control: `services.configs.<name>.embeddings.api_key` is the same field, nested.

    `_parse_embeddings` (and its call to `_resolve_env_var`) runs for a
    per-service `embeddings:` section too, so the substitution must reach
    there as well, not only the top-level `embeddings:` key.
    """
    fragment = (
        "services:\n"
        "  data_dir: ./data\n"
        "  configs:\n"
        "    main:\n"
        "      embeddings:\n"
        "        provider: openai-compatible\n"
        '        api_key: "${OPENAI_API_KEY}"\n'
    )
    os.environ.pop("OPENAI_API_KEY", None)
    assert block_config_error(fragment, complete=False) is None


def test_checker_does_not_repair_invalid_yaml_via_substitution() -> None:
    """Regression: substituting in the text let a genuinely broken flow mapping parse.

    `${DOCS_DB}` contains `{`/`}`, which are flow-collection indicators in a
    YAML flow mapping. `yaml.safe_load` rejects this document as written,
    regardless of what the environment holds -- substitution must happen
    only after a successful parse, never before it.
    """
    error = block_config_error("database: {path: ${DOCS_DB}}\n", complete=True)
    assert error is not None
    assert "invalid YAML" in error.message


def test_checker_does_not_let_a_placeholder_complete_a_deleted_database_section() -> None:
    """Regression: a complete block with its `database:` section removed is not repaired.

    Deleting `docs/configuration.md`'s "Create a `engrava.yaml` file:"
    example's `database:` section is a real documentation defect (the file it
    tells the reader to create would not load) -- injecting a placeholder
    path for it would hide exactly that.
    """
    complete_block = _unique_block("docs/configuration.md", "wal_mode: true")
    without_database = complete_block.body.split("search:", 1)[1]
    mutated = "search:" + without_database
    assert "database" not in yaml.safe_load(mutated)

    error = block_config_error(mutated, complete=True)
    assert error is not None
    assert "database.path" in error.message


def test_checker_distinguishes_an_empty_document_from_an_explicit_null() -> None:
    """Regression: an empty block and a `null` block must not share a verdict.

    A truly empty block has nothing to check. A block whose content is the
    literal `null` is a real value a reader's own parser also produces
    `None` for, and `load_config` rejects a non-mapping document.
    """
    assert block_config_error("", complete=False) is None
    assert block_config_error("   \n  \n", complete=False) is None

    null_error = block_config_error("null\n", complete=False)
    assert null_error is not None
    assert "mapping" in null_error.message


def test_checker_rejects_a_duplicate_key_that_discards_an_invalid_value() -> None:
    """Regression: a duplicate key must not let plain YAML silently keep only the last.

    `wal_mode: "not-a-boolean"` immediately before the real `wal_mode: true`
    used to pass, because `yaml.safe_load` keeps only the second occurrence
    -- discarding a shown value unread instead of judging it.
    """
    fragment = 'database:\n  path: demo.db\n  wal_mode: "not-a-boolean"\n  wal_mode: true\n'
    error = block_config_error(fragment, complete=True)
    assert error is not None
    assert "duplicate key" in error.message


def test_checker_judges_a_database_fragment_missing_only_path() -> None:
    """Regression: `database: {wal_mode: true}` as a fragment is judged on what it shows.

    Injection used to handle a missing `database` section but not a present
    one missing `path`, so this fragment failed with `'database.path' is
    required` -- a defect not shown by the fragment at all.
    """
    assert block_config_error("database:\n  wal_mode: true\n", complete=False) is None


def test_checker_accepts_a_yaml_merge_key() -> None:
    """Regression: `<<:` is a merge directive, not a duplicate (or invalid) key.

    A naive check must not construct the merge key's own node as if it were
    a plain key (it carries a special tag with no registered constructor
    outside `flatten_mapping`'s own handling of it), and must not treat the
    key it introduces as a duplicate just because `flatten_mapping` folds it
    in under the same name as another key already present.
    """
    fragment = "database:\n  <<: {path: demo.db}\n  wal_mode: true\n"
    assert block_config_error(fragment, complete=True) is None


def test_checker_accepts_an_explicit_key_that_overrides_a_merged_one() -> None:
    """Regression: an explicit key beats a merge-introduced one; that is not a duplicate.

    Checking the *flattened* result (after the merge key's keys are folded
    in) cannot tell "written twice" from "one written, one introduced by
    merge precedence" -- it sees `path` twice either way. The real loader
    resolves this to the explicit value (`demo.db`) without complaint, so
    the check must too.
    """
    fragment = "database:\n  <<: {path: base.db}\n  path: demo.db\n"
    assert block_config_error(fragment, complete=True) is None


def test_checker_still_rejects_a_duplicate_key_inside_a_merged_mapping() -> None:
    """Control: a genuine duplicate inside the mapping a merge key references is still caught.

    That mapping is its own node, constructed (and checked) independently of
    the mapping doing the merging -- fixing the false positive above must
    not also blind the check to a real duplicate one level down.
    """
    fragment = "x: &defaults\n  path: a.db\n  path: b.db\ndatabase:\n  <<: *defaults\n"
    error = block_config_error(fragment, complete=True)
    assert error is not None
    assert "duplicate key" in error.message


def test_checker_accepts_overlapping_keys_across_a_merge_sequence() -> None:
    """Control: two merge sources sharing a key is precedence, not a duplicate.

    `<<: [a, b]` merges a list of mappings; YAML resolves a key present in
    more than one source by precedence (the first one wins), not by error.
    Each source mapping has the shared key written only once in its own
    list, which is what this check inspects -- it never compares keys
    *across* different mapping nodes.
    """
    fragment = "database:\n  <<: [{path: a.db}, {path: b.db}]\n"
    assert block_config_error(fragment, complete=True) is None
