"""Structural checks for the shipped ``examples/mcp-client-config*.json`` files.

These files are meant to be copied verbatim into a real MCP client's own
config file, so their *shape* -- not just their JSON validity -- is the
contract: a client that reads a different top-level key than the one a file
actually has silently starts no server at all, with no parse error to flag
it. VS Code in particular has two shapes that both exist in the wild --
``.vscode/mcp.json`` (and the profile-level ``mcp.json``) put ``servers`` at
the top level, while the older *settings* form (multi-root workspace
settings, and the user ``settings.json`` VS Code auto-migrates away) nests
the same object one level deeper under an ``mcp`` key. The shipped VS Code
example must be the former, since that is the file ``examples/README.md``
tells a reader to copy it into.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# tests/examples/test_mcp_client_configs.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"

MCP_CLIENT_CONFIG_FILES: tuple[Path, ...] = tuple(
    sorted(EXAMPLES_DIR.glob("mcp-client-config*.json"))
)

VSCODE_CONFIG = EXAMPLES_DIR / "mcp-client-config.vscode.json"


def test_at_least_one_mcp_client_config_file_ships() -> None:
    """Pin the glob's non-emptiness, so a rename cannot make every test below vacuous."""
    assert MCP_CLIENT_CONFIG_FILES, (
        f"no examples/mcp-client-config*.json files found under {EXAMPLES_DIR}"
    )
    assert VSCODE_CONFIG in MCP_CLIENT_CONFIG_FILES, f"{VSCODE_CONFIG.name} does not exist"


def test_every_mcp_client_config_file_parses_as_json() -> None:
    """Every shipped file is valid JSON, loadable as-is by a real client."""
    for path in MCP_CLIENT_CONFIG_FILES:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{path.name} is not valid JSON: {exc}")


def test_every_server_block_has_a_command() -> None:
    """Every server entry, in every file, names the executable an MCP client would launch."""
    for path in MCP_CLIENT_CONFIG_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        servers = data["servers"] if path == VSCODE_CONFIG else data["mcpServers"]
        for name, block in servers.items():
            assert "command" in block, f"{path.name}: server {name!r} has no 'command' key"


def test_vscode_config_has_servers_at_top_level_and_no_mcp_wrapper() -> None:
    """The VS Code file is the ``.vscode/mcp.json`` shape, not the settings-form wrapper.

    Copied as the settings-form wrapper into ``mcp.json``, the file defines
    no server under the key VS Code actually reads there, and VS Code never
    starts it.
    """
    data = json.loads(VSCODE_CONFIG.read_text(encoding="utf-8"))
    assert "servers" in data, f"{VSCODE_CONFIG.name} has no top-level 'servers' key"
    assert "mcp" not in data, (
        f"{VSCODE_CONFIG.name} has a top-level 'mcp' key -- that is the settings-form "
        "wrapper, not the .vscode/mcp.json shape this file is meant to be copied into"
    )


def test_non_vscode_configs_keep_mcpservers_top_level() -> None:
    """Every other client config keeps the ``mcpServers`` shape (Claude Desktop, Cursor, ...)."""
    for path in MCP_CLIENT_CONFIG_FILES:
        if path == VSCODE_CONFIG:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "mcpServers" in data, f"{path.name} has no top-level 'mcpServers' key"
        assert "servers" not in data, f"{path.name} unexpectedly has a top-level 'servers' key"
