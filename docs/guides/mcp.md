# MCP server

Engrava ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that exposes a memory store to any MCP-capable client — Claude Desktop,
Claude Code, Cursor, Windsurf, VS Code, and others. Point a client at it and the
assistant can search, read, and (optionally) write the same engrava store your
application uses, with no glue code.

The server is an *API consumer*, not an engrava extension: it wraps engrava's
public async API and speaks MCP over **stdio**. Think of it as a sibling of the
[CLI](../cli.md) that talks to MCP clients instead of a terminal.

> New to the model (thought, edge, reflection, cycle)? Read
> [Core Concepts](../concepts.md) first — this guide uses those terms.

## Install

The server lives behind the `mcp` extra:

```bash
pip install "engrava[mcp]"
```

The extra pulls the MCP SDK and its transport stack. It installs **only** with
the extra — plain `pip install engrava` is unaffected and stays dependency-light,
so applications that embed engrava as a library never pay for the server they
do not run.

## Run

The server is a standalone process served over stdio. Two equivalent entry
points are installed with the extra:

```bash
engrava-mcp
```

```bash
python -m engrava.mcp
```

Both build the same server and serve it on stdio. You normally do **not** launch
it by hand — an MCP client spawns it as a subprocess using one of these commands
(see [Client configuration](#client-configuration) below). Running it directly
in a terminal is mostly useful for a quick smoke test; it will wait for an MCP
client to speak to it over stdin and exits on EOF.

### Pointing the server at a store

The server resolves its store from the environment when it starts. Two variables
are recognised, in priority order:

| Variable | Value | Effect |
|---|---|---|
| `ENGRAVA_MCP_CONFIG` | Path to an `engrava.yaml` | Builds the store with the configured embedding provider, vector backend, journal, and TTL settings (`SqliteEngravaCore.from_config`). |
| `ENGRAVA_DB_PATH` | Path to a SQLite database file | Opens that file directly and ensures the schema. No embedding provider or vector backend is configured, so hybrid search degrades to its lexical backend. |

`ENGRAVA_MCP_CONFIG` takes precedence: if both are set, the config file wins. If
**neither** is set, the server has no store to serve and tool calls return an
actionable error telling you to set one of the two variables.

Use `ENGRAVA_MCP_CONFIG` whenever you want semantic (vector) search or any
non-default storage settings; the database created by your application via
[`engrava.yaml`](../configuration.md) is the same file the server should open.
Use `ENGRAVA_DB_PATH` for a quick lexical-only connection to a bare database
file.

```bash
# Full configuration — semantic search, journal, TTL, etc.
export ENGRAVA_MCP_CONFIG=/path/to/engrava.yaml
engrava-mcp

# Or a bare database file — lexical search only
export ENGRAVA_DB_PATH=/path/to/agent-memory.db
engrava-mcp
```

In a client configuration these become `env` entries on the server block, shown
next.

## Client configuration

Every MCP client that speaks stdio uses the same `mcpServers` shape: a command,
its arguments, and an environment block. Engrava is a **native stdio server**, so
clients spawn `engrava-mcp` directly. There is no HTTP endpoint to host and —
unlike HTTP-only MCP servers — **no `npx mcp-remote` shim** to wedge between the
client and the server. Fewer moving parts, one process, local by default.

A ready-to-copy sample for each client below lives in
[`examples/`](https://github.com/sovantica/engrava/blob/main/examples). Replace
the `ENGRAVA_MCP_CONFIG` path (or swap it for `ENGRAVA_DB_PATH`) with your own
store.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "engrava": {
      "command": "engrava-mcp",
      "env": {
        "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml"
      }
    }
  }
}
```

Restart Claude Desktop; "engrava" appears in the tools menu.

### Claude Code

Register the server from the project root:

```bash
claude mcp add engrava --env ENGRAVA_MCP_CONFIG=/absolute/path/to/engrava.yaml -- engrava-mcp
```

That writes an `mcpServers` entry of the same shape into your Claude Code
configuration. Equivalent JSON, if you prefer to edit it directly:

```json
{
  "mcpServers": {
    "engrava": {
      "command": "engrava-mcp",
      "env": {
        "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml"
      }
    }
  }
}
```

### Cursor

Add an entry to `.cursor/mcp.json` (project-scoped) or the global
`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "engrava": {
      "command": "engrava-mcp",
      "env": {
        "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml"
      }
    }
  }
}
```

### Windsurf

Add an entry to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "engrava": {
      "command": "engrava-mcp",
      "env": {
        "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml"
      }
    }
  }
}
```

### VS Code

VS Code's MCP support nests the servers under an `mcp` key. Add this to your
user `settings.json` or a workspace `.vscode/mcp.json`:

```json
{
  "mcp": {
    "servers": {
      "engrava": {
        "command": "engrava-mcp",
        "env": {
          "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml"
        }
      }
    }
  }
}
```

> **Other clients.** Cline, Codex, and most other stdio MCP clients use the same
> `command` / `args` / `env` block as Claude Desktop and Cursor above — copy any
> of those entries. If a client cannot find `engrava-mcp` on its `PATH`, set
> `command` to the absolute path of the script inside your virtual environment
> (for example `/path/to/.venv/bin/engrava-mcp`), or use
> `"command": "python"` with `"args": ["-m", "engrava.mcp"]`.

## Tool reference

The server registers **eleven tools**: six read tools that are always available,
and five write tools that are available unless the server is started in
[read-only mode](#read-only-mode). Tools return JSON; thought and edge mutations
return the key fields of the affected record.

Tools carry MCP *annotations* so clients can present them safely: the read tools
are marked read-only, the write tools are marked as writes, and the two `delete`
tools additionally carry a **destructive** hint so a client can warn before it
runs them.

### Read tools (always available)

| Tool | Purpose | Key arguments |
|---|---|---|
| `get_thought` | Fetch a single thought by its identifier. Returns a `found` flag and the thought (or `null`). | `thought_id` |
| `search_memory` | Hybrid ranked search (lexical + vector + recency). Returns ranked `thought_id`/`score` pairs and the `backends_used`. | `query_text`; `top_k` (default 10); `include_reflections` (default true); optional `thought_type`, `lifecycle_status`, `priority` |
| `search_keywords` | Pure full-text BM25 keyword search. Supports `AND`, `OR`, `NOT`, and prefix `*`. Returns ranked `thought_id`/`score` pairs. | `query`; `top_k` (default 10) |
| `list_memory` | Deterministic, unranked browse over stored thoughts — no score, newest first. The home for "list memory by structured field". | `thought_type`, `lifecycle_status`, `priority`, `min_cycle`, `max_cycle`, `include_expired` (default false); `limit` (default 50), `offset` (default 0) |
| `query_memory` | Run a structured MindQL `FIND` query, e.g. `FIND thoughts WHERE lifecycle_status = 'ACTIVE' LIMIT 10`. **Only `FIND`** is accepted. Returns `columns` and `rows`. | `query`; optional `limit` (overrides any `LIMIT` in the query) |
| `memory_stats` | Aggregate counts and store-health metrics: thought and edge counts (by type/status) and total storage size. | *(none)* |

A note on `search_memory` filters: the hybrid ranker itself cannot filter, so a
supplied `thought_type` / `lifecycle_status` / `priority` is applied **after**
ranking. A filtered call may therefore return fewer than `top_k` results and adds
a `filtered` block reporting how many ranked hits were `scanned`, `matched`, and
`dropped` — so a short list is never mistaken for "nothing was found". When you
want an exhaustive, paginated listing by those same fields, use `list_memory`
instead.

`query_memory` deliberately accepts **only** the MindQL `FIND` command; raw-SQL
passthrough (`SELECT`), aggregate `COUNT`, and any extension commands are
rejected over the wire. See [MindQL](../mindql.md) for the `FIND` grammar.

### Write tools (hidden in read-only mode)

| Tool | Purpose | Key arguments | Annotation |
|---|---|---|---|
| `store_thought` | Create a new thought node. New thoughts start in `CREATED` lifecycle state. Returns the created thought's identifier and key fields. | `essence`, `content`; optional `thought_type` (default `NOTE`), `priority` (default `P3`), `source` (default `"agent"`), `confidence`, `thought_id`, `deduplicate` | write |
| `update_thought` | Update selected fields of an existing thought. Only supplied fields change; the rest are untouched. | `thought_id`; optional `essence`, `content`, `priority`, `lifecycle_status`, `confidence` | write (idempotent) |
| `link_thoughts` | Create a typed edge between two existing thoughts. Both endpoints must already exist. | `from_thought_id`, `to_thought_id`, `edge_type`; optional `weight` (default 1.0), `edge_id` | write |
| `delete_thought` | Delete a thought by its identifier. Deleting an absent id is a no-op (returns `deleted: false`), not an error. | `thought_id` | **destructive** |
| `delete_edge` | Delete an edge by its identifier. Deleting an absent id is a no-op (returns `deleted: false`), not an error. | `edge_id` | **destructive** |

`link_thoughts` edges are unique per `(source, target, type)`: linking the same
pair with the same type twice is rejected rather than ignored, so this write is
not idempotent. The valid `thought_type`, `lifecycle_status`, `priority`, and
`edge_type` values are the engrava enums documented in the
[Glossary](../glossary.md) and [Core Concepts](../concepts.md) (for example
`thought_type` is one of `TASK`, `OBSERVATION`, `BELIEF`, `REFLECTION`,
`OUTPUT_DRAFT`, `NOTE`).

## Resources

Where tools are *invoked*, **resources** are addressable `engrava://` URIs that a
client surfaces as attachable context (drop them into a conversation, no tool
call). Three resources are registered. They are reads by definition, so they are
**always available** — they are *not* hidden by read-only mode — and each returns
a JSON document (`application/json`).

| Resource | Returns |
|---|---|
| `engrava://thought/{thought_id}` | A single thought as JSON. Reading an unknown identifier yields a graceful not-found payload rather than an error. |
| `engrava://stats` | Store-health counts and total size — the same payload as the `memory_stats` tool (both share one implementation, so they always agree). |
| `engrava://recent` | The most-recently-updated thoughts (newest first) as JSON. |

## Prompts

**Prompts** are parameterised templates a client surfaces as slash-commands or
buttons. Each renders a ready-to-send instruction that guides the assistant to
gather context with the read tools and resources above before answering. They
are templates only — they open no write path and call no mutation. Like
resources, prompts are read-oriented and are **always available**, including in
read-only mode.

| Prompt | What it scaffolds | Arguments |
|---|---|---|
| `summarize_recent_memory` | A concise summary of the most recently stored thoughts, highlighting themes and anything unresolved. | optional `limit` — how many recent thoughts to consider (default 5) |
| `find_related` | Gather and synthesise stored thoughts related to a topic, grouping related points and noting gaps or contradictions. | required `topic` |
| `reflect_on_topic` | A structured reflection over what memory holds about a topic: what is established, open questions, and tensions, with concrete follow-ups. | required `topic` |

## Read-only mode

Set `ENGRAVA_MCP_READ_ONLY` to a truthy value (`1`, `true`, or `yes`,
case-insensitive) to start the server with a **retrieval-only** surface:

```bash
export ENGRAVA_MCP_READ_ONLY=true
export ENGRAVA_MCP_CONFIG=/absolute/path/to/engrava.yaml
engrava-mcp
```

In read-only mode the five write tools — `store_thought`, `update_thought`,
`link_thoughts`, `delete_thought`, `delete_edge` — are **not registered at all**,
so they are never advertised to the client. The six read tools, all three
resources, and all three prompts remain available.

Use it for any deployment that should only retrieve from memory and must not be
able to change it — a shared read-only store, a demo, an analytics or
question-answering client. Because the write tools are absent rather than merely
guarded, a client in read-only mode has no path to mutate the store.

As an `env` block on a server entry (any client):

```json
{
  "mcpServers": {
    "engrava": {
      "command": "engrava-mcp",
      "env": {
        "ENGRAVA_MCP_CONFIG": "/absolute/path/to/engrava.yaml",
        "ENGRAVA_MCP_READ_ONLY": "true"
      }
    }
  }
}
```

## Optional vector search

Hybrid search (`search_memory`) combines lexical, vector, and recency signals.
Whether the **vector** signal is active depends on how the store is configured:

- With `ENGRAVA_MCP_CONFIG` pointing at an `engrava.yaml` that configures an
  embedding provider, `search_memory` uses semantic vectors. Installing the
  [`vec` extra](../search.md) (`pip install "engrava[vec,mcp]"`) adds the
  `sqlite-vec` backend for faster KNN; without it the vector signal still works
  via the built-in numpy backend.
- With `ENGRAVA_DB_PATH` (a bare database file) — or any store without an
  embedding provider — there is no vector signal, and `search_memory` degrades
  gracefully to lexical (BM25) ranking. `search_keywords` is pure BM25 either
  way.

The `backends_used` field on a `search_memory` response tells you which signals
actually contributed to a given query, so you can confirm whether vectors were in
play.

## Notes

- The server is **single-writer**, like engrava itself — point it at a store that
  is not being written concurrently by another process (see
  [Concurrency](../concurrency.md)).
- Tool errors are returned as clean, actionable messages (for example, an unknown
  `thought_id`, or a non-`FIND` query) rather than raw tracebacks.
- Thoughts and edges created through the write tools start at cycle `0`: this API
  consumer has no notion of the agent [cycle clock](../concepts.md#cycle-the-agent-clock),
  which your application owns.

## Next

- [Core Concepts](../concepts.md) — thought / edge / reflection / cycle.
- [Hybrid Search](../search.md) — how the retrieval ranking works.
- [MindQL](../mindql.md) — the `FIND` grammar behind `query_memory`.
- [Configuration](../configuration.md) — wiring an embedding provider via `engrava.yaml`.
- [CLI Reference](../cli.md) — the terminal-facing sibling of this server.
