"""Model Context Protocol server for engrava.

Exposes engrava's public read API as MCP tools over stdio, so MCP-aware
agents can fetch thoughts, search memory, and run structured queries.
The server is a standalone API consumer — it registers no engrava hooks,
manifests, or extensions.

Run it with ``python -m engrava.mcp`` or the ``engrava-mcp`` console
script.
"""

from __future__ import annotations

from engrava.mcp.server import build_server, main

__all__ = ["build_server", "main"]
