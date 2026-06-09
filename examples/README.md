# engrava examples

Runnable scripts that exercise the engine end-to-end on a small demo
dataset. The scripts are standalone — no extra setup, no external
services, no API keys.

## Prerequisites

```bash
pip install 'engrava[embeddings-local]'
```

The `embeddings-local` extra pulls `sentence-transformers` and `torch`
and downloads a small (~30-90 MB) encoder model on first use. The
encoder is a vector-producing model, **not** a language model — no
API keys are needed and there is no network traffic after the first
download.

## What is here

| Script | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | 5-minute end-to-end tour: in-memory store, percepts + utterances ingest, one dreaming cycle, hybrid-search query, top-K print. |
| [`simple_agent.py`](simple_agent.py) | Lower-level walkthrough using a custom scoring hook, manual edges, and a fake embedding function — useful for understanding the API surface without the local-encoder dependency. |

## MCP client configuration

Sample `mcpServers` blocks for pointing an MCP client (Claude Desktop, Claude
Code, Cursor, Windsurf, VS Code, …) at the engrava
[MCP server](../docs/guides/mcp.md). Copy the one that matches your client and
replace the store path with your own.

| File | For |
|---|---|
| [`mcp-client-config.json`](mcp-client-config.json) | The default stdio block (Claude Desktop, Claude Code, Cursor, Windsurf, Cline, Codex, …). Points at an `engrava.yaml`. |
| [`mcp-client-config.db-path.json`](mcp-client-config.db-path.json) | Same shape, but points at a bare SQLite file via `ENGRAVA_DB_PATH` (lexical search only). |
| [`mcp-client-config.readonly.json`](mcp-client-config.readonly.json) | Read-only deployment — `ENGRAVA_MCP_READ_ONLY=true` hides the write tools. |
| [`mcp-client-config.vscode.json`](mcp-client-config.vscode.json) | VS Code, which nests servers under an `mcp` key. |

These require the MCP extra: `pip install "engrava[mcp]"`.

Run them directly with the Python interpreter:

```bash
python examples/quickstart.py
python examples/simple_agent.py
```

To see the dreaming consolidation step actually produce REFLECTION
nodes, run the bundled synthetic benchmark on a representative
workload (it builds a multi-conversation corpus where memories
accumulate and repeat — the conditions dreaming is built for):

```bash
python -m engrava.benchmarks.synthetic
```

## Notes on output

The hybrid-search scores are produced by floating-point arithmetic on
the encoder output; absolute score values depend on the loaded model
version and on the host hardware. The retrieved ordering on the
shipped demo dataset is stable: the top result for the quickstart
query (`What is the user's favorite color?`) is `My favorite color is
teal.`.

## Further reading

- [`docs/quickstart.md`](../docs/quickstart.md) — narrative walkthrough
  paired with `quickstart.py`.
- [`docs/dreaming.md`](../docs/dreaming.md) — what dreaming does and
  how to configure it.
- [`docs/benchmarks.md`](../docs/benchmarks.md) — the synthetic
  benchmark suite that reports dreaming's measured REFLECTION
  coverage on a representative workload.
- [`docs/guides/mcp.md`](../docs/guides/mcp.md) — the MCP server:
  install, run, client configuration, and the full
  tool/resource/prompt reference.
