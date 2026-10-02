# engrava examples

Runnable scripts that exercise the engine end-to-end on a small demo
dataset. The scripts are standalone — no extra setup, no external
services, no API keys.

## Prerequisites

```bash
pip install 'engrava[embeddings-local]'
```

The `embeddings-local` extra pulls `sentence-transformers` and `torch`
— a large one-time download (~550+ MB on the PyPI Linux/x86_64 wheel
for Python 3.11) — plus a further encoder model download on first use.
See [`docs/configuration.md` → Quick-start profiles](../docs/configuration.md#quick-start-profiles)
for the exact, platform-scoped numbers. The encoder is a
vector-producing model, **not** a language model — no API keys are
needed, and — once `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set —
there is no network traffic after the first download (see
[`docs/configuration.md` → `local`](../docs/configuration.md#local--sentence-transformers-offline-after-warm-up)
for why those two variables are needed).

## What is here

| Script | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | 5-minute end-to-end tour: in-memory store, percepts + utterances ingest, one dreaming cycle, hybrid-search query, top-K print. |
| [`notes_memory.py`](notes_memory.py) | The runnable companion to [`docs/tutorial.md`](../docs/tutorial.md): ingests a handful of notes, links related ones with an edge, and searches them, using a deterministic hash in place of a real embedding model. |
| [`simple_agent.py`](simple_agent.py) | Lower-level walkthrough using a custom scoring hook, manual edges, and a fake embedding function — useful for understanding the API surface without the local-encoder dependency. |
| [`agent_loop.py`](agent_loop.py) | A per-turn memory-backed agent loop: store the message, retrieve prior context with hybrid search, call an LLM (a canned stand-in here), store the reply, and run dreaming consolidation every few turns. |

## Quick-start configuration profiles

Three copy-ready `engrava.yaml` files, one per install path — pick the one
that matches how much you want a machine-learning model in your process:

| File | Profile | What you get |
|---|---|---|
| [`profile-lexical.yaml`](profile-lexical.yaml) | `lexical` | Keyword search only. No embeddings dependency, no download. |
| [`profile-network-ollama.yaml`](profile-network-ollama.yaml) | `network` | Semantic search via a running Ollama server — the model stays out of this process. |
| [`profile-local.yaml`](profile-local.yaml) | `local` | Semantic search in-process, offline after a one-time download. |

See [`docs/configuration.md` → Quick-start profiles](../docs/configuration.md#quick-start-profiles)
for the install command, the real download-size numbers, and what each
profile trades off against the others.

[`config.yaml`](config.yaml) is a different kind of file: not a fourth
profile, but a fuller reference config with dreaming consolidation, the
edge-creation gates, and every hybrid-search weight spelled out — a
starting point to copy and tune once you have picked a profile above.

## MCP client configuration

Sample `mcpServers` blocks for pointing an MCP client (Claude Desktop, Claude
Code, Cursor, Windsurf, VS Code, …) at the engrava
[MCP server](https://github.com/sovantica/engrava-mcp). Copy the one that matches your client and
replace the store path with your own.

| File | For |
|---|---|
| [`mcp-client-config.json`](mcp-client-config.json) | The default stdio block (Claude Desktop, Claude Code, Cursor, Windsurf, Cline, Codex, …). Points at an `engrava.yaml`. |
| [`mcp-client-config.db-path.json`](mcp-client-config.db-path.json) | Same shape, but points at a bare SQLite file via `ENGRAVA_DB_PATH` (lexical search only). |
| [`mcp-client-config.readonly.json`](mcp-client-config.readonly.json) | Read-only deployment — `ENGRAVA_MCP_READ_ONLY=true` hides the write tools. |
| [`mcp-client-config.vscode.json`](mcp-client-config.vscode.json) | VS Code, which nests servers under an `mcp` key. |

These drive the standalone [`engrava-mcp`](https://github.com/sovantica/engrava-mcp) server — install it with `uvx engrava-mcp` (or `pip install engrava-mcp`).

Run them directly with the Python interpreter:

```bash
python examples/quickstart.py
python examples/notes_memory.py
python examples/simple_agent.py
python examples/agent_loop.py
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
- [`docs/tutorial.md`](../docs/tutorial.md) — narrative walkthrough
  paired with `notes_memory.py`.
- [`docs/dreaming.md`](../docs/dreaming.md) — what dreaming does and
  how to configure it.
- [`docs/benchmarks.md`](../docs/benchmarks.md) — the synthetic
  benchmark suite that reports dreaming's measured REFLECTION
  coverage on a representative workload.
- [`engrava-mcp`](https://github.com/sovantica/engrava-mcp) — the standalone
  MCP server: install, run, client configuration, and the full
  tool/resource/prompt reference.
