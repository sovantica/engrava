"""Layer 7 of the documentation-example tests — ``yaml`` config vs the real loader.

This layer extracts the documented ``yaml`` fenced blocks and checks them,
using ``load_config`` for the key names and values, except the blocks
registered as exempt.

Delegating to the real loader, not reimplementing it
------------------------------------------------------
Key names and values are validated by ``load_config`` itself; this module
keeps no mirrored key or value model. The raw document stays intact until it
is a real Python value:

0. **Check for a duplicate written key on the composed node graph, before
   anything is constructed.** A plain ``yaml.safe_load`` silently keeps only
   the last of a duplicate key. "Is this key written twice in this mapping"
   is a property of the document's *text*, not of a constructed value, so
   ``yaml.compose()`` gets the node graph (composed, not constructed — no
   construction and no merge flattening; ``yaml.compose()`` does still
   resolve an alias to the *same* mapping node object its anchor produced,
   which is exactly what lets a duplicate inside a shared, aliased mapping
   still be caught) and ``_find_duplicate_key`` walks it directly: for each
   ``MappingNode``, its own written keys, skipping the merge key itself. See
   that function's docstring for the merge-key and alias cases.
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

Fragment vs. complete, honestly
--------------------------------
The documented ``yaml`` blocks listed in ``COMPLETE_YAML_BLOCKS`` are
registered as complete. The other checkable blocks are treated as fragments.
This is not a defect to fix by reclassifying more blocks as complete — a
snippet titled "Configure it via ``SearchConfig``:" or "A minimal enable:" is
genuinely illustrating one section, not publishing a runnable file.

Two documented blocks each show the same top-level key (``manifests:``)
twice, side by side, to illustrate alternate forms. That is a real duplicate
key by construction, so the duplicate-key check (correctly) reports it —
checking either whole block as one document can only ever report the
duplicate, never validate either form on its own. Both blocks are
registered as exempt (``ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS``)
rather than silently under-checked.
"""

from __future__ import annotations

import dataclasses
import os
import re
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
    block_digest,
    exemption_digest_problems,
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
#
# Each entry is (file, anchor, reason, digest). The anchor finds the block; the
# digest (block_digest of the block's text) binds the exemption to the text it
# was granted for. The exemption applies only while the block still matches its
# digest: edit an exempt block and it goes back to being checked, and
# test_exempt_yaml_registry_digests_match names it and prints the digest to
# register if the edit was deliberate.
EXEMPT_YAML_BLOCKS: tuple[tuple[str, str, ExemptionReason, str], ...] = (
    (
        "docs/configuration.md",
        "# list form",
        ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS,
        "91d1c03d4bfe3f07",
    ),
    (
        "docs/extensions.md",
        "# Explicit dotted paths",
        ExemptionReason.DUPLICATE_KEY_ALTERNATE_FORMS,
        "7b9ffbfa2ac75990",
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
    ("docs/configuration.md", "vec0_overfetch_factor: 4"),
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
    constructed.

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


def _registered_exemptions() -> list[tuple[CodeBlock, ExemptionReason, str]]:
    """Resolve every EXEMPT_YAML_BLOCKS entry to (block, reason, registered digest)."""
    return [
        (_unique_block(rel, anchor), reason, digest)
        for rel, anchor, reason, digest in EXEMPT_YAML_BLOCKS
    ]


def _exempt_locations() -> dict[str, ExemptionReason]:
    """Locations of the blocks that are exempt *now*.

    An entry counts only while its block's text still matches the registered
    digest. An edited block drops out of this mapping, so it is checked like
    any other block; the digest census reports it by name.
    """
    return {
        block.location: reason
        for block, reason, digest in _registered_exemptions()
        if block_digest(block.body) == digest
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
    _registered_exemptions()


def test_exempt_yaml_registry_digests_match() -> None:
    """Every exempt block still has the text its exemption was granted for.

    The text is compared with its line endings normalised. An edited block is
    checked again (see ``_exempt_locations``) and is named here with the digest
    to register if the edit was deliberate.
    """
    problems = exemption_digest_problems(
        "EXEMPT_YAML_BLOCKS",
        [(block, digest) for block, _, digest in _registered_exemptions()],
    )
    assert not problems, "exempt yaml blocks were edited:\n" + "\n".join(problems)


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
    does and does not prove). An exempt block is skipped only while its text
    still matches the digest its exemption was registered with.
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
    """`hooks: null` is accepted by the real loader (means "unset")."""
    assert block_config_error("hooks: null\n", complete=False) is None


def test_checker_does_not_let_a_column_zero_comment_split_a_mapping() -> None:
    """A comment between two keys of one mapping is not a new document."""
    fragment = (
        "search:\n"
        "  default_fts_weight: 0.30\n"
        "\n"
        "# More search settings\n"
        "  default_vector_weight: 0.55\n"
    )
    assert block_config_error(fragment, complete=False) is None


def test_checker_accepts_an_anchor_and_alias_spanning_two_sections() -> None:
    """A YAML anchor defined in one section and used in another works."""
    fragment = "database:\n  path: &db_path demo.db\n\nservices:\n  data_dir: *db_path\n"
    assert block_config_error(fragment, complete=True) is None


def test_checker_rejects_a_non_boolean_wal_mode() -> None:
    """The real loader rejects a non-boolean `wal_mode`."""
    error = block_config_error(
        'database:\n  path: demo.db\n  wal_mode: "yes"\n',
        complete=True,
    )
    assert error is not None
    assert "wal_mode" in error.message


def test_checker_rejects_a_negative_vector_dimension() -> None:
    """The real loader rejects a negative dimension."""
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
    """Substitution must not repair a field the real loader never interpolates.

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
    """Substituting a placeholder must not repair a genuinely broken flow mapping.

    `${DOCS_DB}` contains `{`/`}`, which are flow-collection indicators in a
    YAML flow mapping. `yaml.safe_load` rejects this document as written,
    regardless of what the environment holds -- substitution must happen
    only after a successful parse, never before it.
    """
    error = block_config_error("database: {path: ${DOCS_DB}}\n", complete=True)
    assert error is not None
    assert "invalid YAML" in error.message


def test_checker_does_not_let_a_placeholder_complete_a_deleted_database_section() -> None:
    """A complete block with its `database:` section removed is not repaired.

    Deleting `docs/configuration.md`'s "Create a `engrava.yaml` file:"
    example's `database:` section is a real documentation defect (the file it
    tells the reader to create would not load) -- injecting a placeholder
    path for it would hide exactly that.
    """
    complete_block = _unique_block("docs/configuration.md", "vec0_overfetch_factor: 4")
    without_database = complete_block.body.split("search:", 1)[1]
    mutated = "search:" + without_database
    assert "database" not in yaml.safe_load(mutated)

    error = block_config_error(mutated, complete=True)
    assert error is not None
    assert "database.path" in error.message


def test_checker_distinguishes_an_empty_document_from_an_explicit_null() -> None:
    """An empty block and a `null` block must not share a verdict.

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
    """A duplicate key must not let plain YAML silently keep only the last.

    `wal_mode: "not-a-boolean"` immediately before the real `wal_mode: true`
    would pass under `yaml.safe_load`, which keeps only the second occurrence
    -- discarding a shown value unread instead of judging it.
    """
    fragment = 'database:\n  path: demo.db\n  wal_mode: "not-a-boolean"\n  wal_mode: true\n'
    error = block_config_error(fragment, complete=True)
    assert error is not None
    assert "duplicate key" in error.message


def test_checker_judges_a_database_fragment_missing_only_path() -> None:
    """A fragment with a `database` section but no `path` gets the placeholder path.

    `database: {wal_mode: true}` is judged on what it shows, so it is not
    reported for a missing `'database.path'` it never shows.
    """
    assert block_config_error("database:\n  wal_mode: true\n", complete=False) is None


def test_checker_accepts_a_yaml_merge_key() -> None:
    """`<<:` is a merge directive, not a duplicate (or invalid) key.

    A naive check must not construct the merge key's own node as if it were
    a plain key (it carries a special tag with no registered constructor
    outside `flatten_mapping`'s own handling of it), and must not treat the
    key it introduces as a duplicate just because `flatten_mapping` folds it
    in under the same name as another key already present.
    """
    fragment = "database:\n  <<: {path: demo.db}\n  wal_mode: true\n"
    assert block_config_error(fragment, complete=True) is None


def test_checker_accepts_an_explicit_key_that_overrides_a_merged_one() -> None:
    """An explicit key beats a merge-introduced one; that is not a duplicate.

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
    the mapping doing the merging -- the merge handling above must not also
    blind the check to a real duplicate one level down.
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


def _edit_registered_block(
    monkeypatch: pytest.MonkeyPatch,
    rel: str,
    anchor: str,
    suffix: str,
) -> CodeBlock:
    """Swap one registered exempt block for a copy with ``suffix`` appended to its text.

    The replacement is made in this module's block list, so the registry, the
    census and the per-block test all see the edited block exactly as they
    would see an edited documentation page.
    """
    original = _unique_block(rel, anchor)
    edited = dataclasses.replace(original, body=original.body + suffix)
    monkeypatch.setitem(
        globals(),
        "_ALL_YAML_BLOCKS",
        [edited if block is original else block for block in _ALL_YAML_BLOCKS],
    )
    return edited


def test_an_edited_exempt_block_is_checked_again_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key appended to an exempt block does not leave the block exempt.

    The exemption was granted for the block's original text, so it no longer
    applies: the per-block test runs (rather than skips) and reports a problem
    with the block, and the digest census names the block and prints the digest
    to register if the edit was deliberate.
    """
    edited = _edit_registered_block(
        monkeypatch, "docs/configuration.md", "# list form", "\nturbo:\n  enabled: true\n"
    )

    assert edited.location not in _exempt_locations()
    with pytest.raises(AssertionError, match="rejected by the real load_config"):
        test_documented_config_keys_exist_on_the_real_classes(edited)

    with pytest.raises(AssertionError, match=re.escape(edited.location)) as census:
        test_exempt_yaml_registry_digests_match()
    message = str(census.value)
    assert block_digest(edited.body) in message
    assert "EXEMPT_YAML_BLOCKS" in message


def test_the_unedited_exempt_blocks_are_exempt() -> None:
    """Control: on the shipped documentation both registered blocks are still exempt."""
    assert len(_exempt_locations()) == len(EXEMPT_YAML_BLOCKS)
