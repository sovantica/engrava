"""Load a workflow file for the tests that pin its wiring, with a strict YAML loader."""

from __future__ import annotations

import re
from typing import Any

import yaml

BOOL_TAG = "tag:yaml.org,2002:bool"
MERGE_TAG = "tag:yaml.org,2002:merge"


class WorkflowLoader(yaml.SafeLoader):
    """Reads a plain ``on`` as a string, and refuses a repeated key and a merge key."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == MERGE_TAG:
                msg = "found a merge key"
                raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                msg = f"found a duplicate key {key!r}"
                raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


WorkflowLoader.yaml_implicit_resolvers = {
    first: [(tag, pattern) for tag, pattern in resolvers if tag != BOOL_TAG]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
WorkflowLoader.add_implicit_resolver(
    BOOL_TAG, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def load_workflow_text(text: str) -> dict[Any, Any]:
    loader = WorkflowLoader(text)
    try:
        loaded = loader.get_single_data()
    finally:
        loader.dispose()
    assert isinstance(loaded, dict)
    return loaded
