# CLI reference

Engrava ships an `engrava` command-line tool for inspecting, querying, and
maintaining a database without writing code. This page documents every command
and option.

```bash
engrava [GLOBAL OPTIONS] COMMAND [ARGS]...
```

## Global options

These apply to every command and go **before** the command name:

| Option | Values / type | Default | Description |
|---|---|---|---|
| `--db` | path | `./engrava.db` | Path to the SQLite database. Falls back to the `ENGRAVA_DB` env var, then the default. |
| `--config` | path | — | Path to `engrava.yaml`. Falls back to the `ENGRAVA_CONFIG` env var. |
| `--format` | `table` \| `json` \| `csv` | `table` | Output format for commands that print records. |
| `--verbose` | flag | off | Emit DEBUG logs from Engrava modules to stderr for this invocation. Command data on stdout keeps its selected format. |
| `--no-extensions` | flag | off | Prevent loading both `engrava.cli` and `engrava.extensions` entry points. `ENGRAVA_DISABLE_EXTENSIONS=1` provides the same control. |
| `--help` | flag | — | Show help and exit (works on the root and on every command). |

**Environment variables.** `ENGRAVA_DB` and `ENGRAVA_CONFIG` are CLI fallbacks for
`--db` and `--config` respectively; the explicit flag always wins
(`--db` > `ENGRAVA_DB` > `./engrava.db`).
`ENGRAVA_DISABLE_EXTENSIONS=1` activates `--no-extensions` before command
resolution, including for root help and built-in commands.

```bash
export ENGRAVA_DB=/data/engrava.db
engrava info                       # uses /data/engrava.db
engrava --db other.db info         # flag overrides the env var
```

## Commands

| Command | Purpose |
|---|---|
| [`info`](#info) | Show a metrics snapshot for the database. |
| [`verify`](#verify) | Verify the audit journal's hash chain. |
| [`query`](#query) | Run a MindQL query. |
| [`remember`](#remember) | Store a thought in one call and print its id. |
| [`recall`](#recall) | Search for thoughts relevant to a query and print ranked results. |
| [`link`](#link) | Create a typed edge between two thoughts. |
| [`snapshot`](#snapshot) | Export thoughts, edges, embeddings, and actions to a JSONL snapshot (not the audit journal). |
| [`restore`](#restore) | Restore a database from a JSONL snapshot. |
| [`gc`](#gc) | Garbage-collect archived thoughts (and optionally expired ones). |
| [`migrate`](#migrate) | Run pending schema migrations. |
| [`export`](#export) | Export thoughts to a portable JSON file. |

## Extension discovery

The executable discovers installed CLI extensions lazily. Built-in commands are
resolved without scanning `engrava.cli`. Root help scans that entry-point group
so installed commands appear in the command list, and an otherwise unknown
command triggers the scan so an installed extension command can resolve.

The built-in `query` command performs a second scan of
`engrava.extensions`. It loads each discovered manifest and registers its
`mindql_extensions` for that query. This query-time scan does not apply manifest
schema migrations.

Both scans import and may execute code from installed packages. Failed entry
points are skipped with a warning log. Put `--no-extensions` before the command,
or set `ENGRAVA_DISABLE_EXTENSIONS=1`, to prevent **both** entry-point groups from
being scanned or loaded:

```bash
engrava --no-extensions --help
ENGRAVA_DISABLE_EXTENSIONS=1 engrava --db memory.db info
engrava --no-extensions --db memory.db query "SELECT thought_id FROM thought LIMIT 5"
```

This control is specific to automatic CLI entry-point loading. It does not
disable explicit manifest paths or library-side `manifests.discover` settings
used by an application-created store. Continue to install only trusted packages;
see
[Security and Trust Boundaries](security.md#extensions-hooks-and-migrations).

`--verbose` enables DEBUG logging for the `engrava` logger hierarchy for the
duration of the command. Logs go to stderr, leaving JSON/CSV/table command data
on stdout.

## Service resolution

The `--service` option on `snapshot` and `restore` resolves the same way in both
commands:

| `--service` | Services config loaded? | Result |
|---|---|---|
| `--service NAME` (explicit) | either | Targets service **NAME**. Its database is looked up in the services `data_dir` if a config is loaded, otherwise in the **parent directory of `--db`** (i.e. `<parent-of-db>/NAME.db`). `snapshot` exits `1` if it does not exist; `restore` creates it. |
| omitted | yes | Falls back to `services.default_service`. |
| omitted | no | Operates on the single `--db` database (not service mode). |

In short: an explicit `--service` works even without a services config (using
`--db`'s directory as the data directory), while omitting it only enters
multi-service mode when a services config is present.

## Schema-version checks

Every built-in command now checks the database's schema version before it
acts, and the two kinds of command are held to different rules:

| Kind | Commands | On a schema below head | On a schema above head |
|---|---|---|---|
| Destructive | `gc`, `restore` | **Refuses**, exit `1`, names `engrava migrate` | **Refuses**, exit `1` |
| Read | `info`, `verify`, `export`, `snapshot`, a `query` that parses as `FIND`/`COUNT`/`SELECT` | Warns on stderr and runs anyway | **Refuses**, exit `1` |

A destructive command never deletes rows through an engine that does not
understand the schema it is deleting from — that gap is how a deleted thought
could come back on an unmigrated database (see
[Deletion on a database that has not been migrated](known-limitations.md#deletion-on-a-database-that-has-not-been-migrated)).
A read command is allowed to attempt anyway, because refusing an ordinary read
over a pending migration would be a worse failure than the one this replaces —
but it is never silent about the gap. `query` classifies by the **parsed**
command, not by the fact that you typed `query`: a registered extension
command can write, so it is refused like a destructive one whenever the
schema is behind.

`migrate` is not in either row — running the pending migrations (or refusing
to, when the database is a populated pre-history schema `ensure_schema()`
cannot safely bootstrap, or is stamped above this build's head version) is
its entire job.

```bash
$ engrava --db old.db gc
Database schema is at version 11; this engrava build's head version is 20. Run 'engrava migrate' before running 'gc' on it.
$ echo $?
1
```

## Unreadable or corrupt databases

`info`, `verify`, `query`, `gc`, `migrate`, and `export` open the configured
database before doing anything else, and so do `snapshot` and `restore` when
no `--config` is also given. A file that exists but is not a valid SQLite
database (truncated, corrupted, or a plain text file), a path that is a
directory, or any other failure while opening it exits `1` with a message
naming the configured path — never a stack trace and never a hang:

```bash
$ engrava --db corrupt.sqlite info
info: corrupt.sqlite: unexpected DatabaseError: file is not a database
$ echo $?
1
```

The named path is exactly the one the invocation was configured with — via
`--db`, `ENGRAVA_DB`, or the CLI's own default — never resolved to an absolute
path the caller did not supply. Rerun with `--verbose` to log the caught
exception's stack (frame filename, line, and function only, the same
deliberately-not-a-full-traceback shape the `remember` / `recall` / `link`
`unexpected_error` exit code uses, below) at `DEBUG` for a bug report.

`snapshot` and `restore` are the two exceptions: when `--config` is also
given to either, the CLI loads it in its own group callback, before either
command's body — and this boundary — ever runs. A `--config` value the
loader itself rejects — missing, unparseable YAML, or the wrong shape —
still exits `1` with a clean `Error: ...` message from that same callback;
a `--config` path that cannot even be opened (a directory, a
permission error, content that is not valid UTF-8) is not caught there and
can exit `1` with a raw traceback instead of a message naming the path.

The `--service` branch of `snapshot` / `restore` resolves a different,
per-service path through service resolution (see above) instead — an
unclassified failure there is not yet named this way either.

### `info`

Shows a metrics snapshot (counts, etc.) for the current database. Takes no
command-specific options.

```bash
engrava --db engrava.db info
# Database: engrava.db
# Metrics schema version: 2 (database schema version: 20)
# Thoughts: 128 ({'OBSERVATION': 100, 'REFLECTION': 28})
# ...
```

The two version numbers are unrelated and are named separately on purpose:
`metrics schema version` is the shape of the `EngravaMetrics` object returned
by `await store.metrics()` (bumped when a field is added to that dataclass),
and `database schema version` is the database's own `PRAGMA user_version` —
the one [Schema-version checks](#schema-version-checks) above gates on.
`--format json info` carries the same two numbers as
`metrics_schema_version` and `database_schema_version`; see the [Upgrade
Guide](upgrade.md#06---07) if you parse that JSON and used to read
`schema_version`.

Use this after an upgrade or a restore to confirm the database is readable and
the counts look right.

### `verify`

Verifies the [audit journal](audit-trail.md)'s hash chain. It walks every
recorded `journal_entry` in sequence order, recomputes each SHA-256 hash, and
checks the parent-hash linkage. Takes no command-specific options. The chain is
verified **regardless of whether journaling is currently enabled**, so a journal
recorded in an earlier session is still auditable.

```bash
engrava --db engrava.db verify
# Journal integrity OK — 128 entries verified.

engrava --db engrava.db --format json verify
# {"valid": true, "entries_checked": 128, ...}
```

The exit code is **`0`** when the chain verifies, **`1`** when it does not (the
text output names the first broken `sequence`, and the JSON output carries
`first_invalid_sequence` / `error_message`) or when the database is missing —
so it drops straight into a CI job, a pre-backup hook, or a monitoring check. An
empty or absent journal verifies as valid with `entries_checked: 0`.

Read the [Security model](audit-trail.md#security-model--guarantees) first: this
is a keyless in-file chain, so it detects accidental corruption and naive edits,
not a chain-aware actor who rewrites the whole `.db`.

### `query`

Executes a [MindQL](mindql.md) query and prints the results in the chosen
`--format`.

```bash
engrava query "MQL"
```

The `MQL` string is a positional argument. It accepts `FIND`, `COUNT`, `SELECT`,
or registered extension commands:

```bash
engrava query "FIND thoughts WHERE lifecycle_status = 'ACTIVE'"
engrava query "COUNT thoughts WHERE priority = 'P1'"
engrava --format json query "SELECT thought_id, essence FROM thought LIMIT 5"
engrava query "FIND thoughts WHERE valid_now"          # only currently-valid facts
```

The bi-temporal `valid_now`, `valid_at`, `valid_within`, and `valid_between`
predicates work here too — see [MindQL](mindql.md) for their full semantics.

## Store resolution (`remember` / `recall` / `link`)

`remember`, `recall`, and `link` are the one-shot memory verbs: store or
search a thought, or create an edge, in a single invocation — no `python -c`
needed. They resolve the database they act on through the same two-tier
precedence, highest first, plus the CLI's own default:

1. A non-empty `--db` (or `ENGRAVA_DB`). This wins outright: with a non-empty
   `--db`, a `--config` file — even a missing or malformed one — is never
   read by these three commands at all. **`--db ""` does not count as given**
   — it is resolved the same way an *omitted* `--db` is (falling through to
   `ENGRAVA_DB`, then the CLI's own default below), because this tier tests
   the value by truthiness, not by whether the flag appeared on the command
   line at all. `--db .` or `--db ./engrava.db` are values; `--db ""` is not.
2. Otherwise, a `--config` file's own `database.path` — loaded the same way
   `SqliteEngravaCore.from_config()` loads it, so a configured `embeddings`,
   `search`, and `journal` section all apply. This is why these three
   commands exist separately from a bare connection: `recall` over a
   configured embedding provider runs the same hybrid search a direct
   library `recall()` call would, vector arm included. A non-empty `--config`
   you named is validated here unconditionally: a file that does not exist,
   or that fails to parse, is always an error (exit `2`) rather than being
   silently treated as though `--config` had never been given. **`--config
   ""` is, by the same truthiness rule as `--db` above, indistinguishable
   from omitting `--config` entirely** — it is never validated, and this
   tier never fires for it.
3. Otherwise, the CLI's own default (`./engrava.db`).

Exactly one tier fires per invocation; `--verbose` reports which one and the
resolved path. None of the three commands takes a `--service` option today
(unlike [`snapshot`](#snapshot) / [`restore`](#restore)), so a service
selected via `services.default_service` is never reachable from any of the
three — that config section is consulted only by `snapshot` / `restore`
themselves.

**Creation.** `remember` and `link` create the resolved database (and its
parent directory) if it does not already exist, printing the path to stderr:

```bash
$ engrava --db new.db remember "first thought"
Created database: new.db
091aa106-fcc0-45a3-a19b-d335ad05ea45
```

`recall` never creates a database — a read against a database nobody wrote to
yet exits `3` naming the resolved path, rather than silently reporting zero
hits:

```bash
$ engrava --db missing.db recall "anything"
Database not found: missing.db
$ echo $?
3
```

**Exit codes**, consistent across all three:

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | An **unanticipated** failure — anything the command's own validation does not specifically check for (a corrupt database file, a directory given as `--db`, an unreadable or non-UTF-8 `--config`, ...). Every one of these three commands runs its whole body under a single error boundary: a check the command performs itself (below) keeps its own specific code and `error` kind, but *any other* exception is converted here instead of tracebacking. The message names the resolved database's path, right after the command name, followed by the exception's own type and text (e.g. `recall: /data/store.db: unexpected DatabaseError: file is not a database`) — actionable, and specific to *this* invocation's database rather than leaving an operator running against several stores to guess which one failed. It is never a stack trace; either half of the exception's own description falls back to a fixed placeholder if it cannot be read safely, and a genuine Ctrl-C or `sys.exit()` raised while that message is being built escapes immediately instead of becoming this exit code at all. The path is only named once the database has actually been resolved — a failure earlier than that (there is none today) would fall back to the path-free `recall: unexpected ...` form. Rerun with `--verbose` to log the caught exception's stack (to stderr, at `DEBUG`) for a real bug report — deliberately not a full traceback: it lists each frame's filename, line number, and function name, read from the exception's own traceback without calling the exception's formatter (or a cause's, a context's, or an exception group's) a second time. That trade gives up some diagnostic detail — no chained-exception text, no source lines, no local variables — for a Ctrl-C or `sys.exit()` landing while `--verbose` builds that output now escaping immediately too, rather than the earlier behaviour where it could be absorbed and the command would still exit `1` with an ordinary error object. |
| `2` | A usage or validation error: an unknown edge type, an empty `TEXT`, a malformed `--meta` / `--filter` token, an out-of-range `--top-k` / `--weight`, or a non-empty `--config` that does not exist or fails to parse. The message names the offending value when available, and for an enum, every valid member. |
| `3` | The resolved database does not exist (`recall` only — `remember` / `link` create it instead). |
| `4` | `link` named a `FROM` or `TO` thought that does not exist. The message names it, when available. |

Exit `2` is also what a small, non-exhaustive set of failures Click itself
catches use — see **What is never JSON** below for why those are plain text
regardless of `--json`, and for the honest (not exhaustive) rule a consumer
should apply instead of expecting a closed list.

**`--json` errors.** Once a `--json` failure reaches the command's own code —
`--type` / `--priority` / `--top-k` / `--weight` already parsed and `--json`
already known — it is *always* a JSON object, never a bare traceback, no
matter what ordinary exception raised it: every specific `error` kind below
keeps its documented exit code, and anything neither this command nor its
libraries were specifically checked for still becomes one, under `"error":
"unexpected_error"`, exit `1` (see the exit-code table above). This does
**not** cover a failure Click's own argument parser rejects before that code
ever runs — a bad flag, an invalid choice, a missing value — which is plain
usage text at exit `2` whether or not `--json` was given, because `--json`
itself has not necessarily been parsed yet; see **What is never JSON** below
for the full, non-exhaustive list. It also does not cover a genuine Ctrl-C or
`sys.exit()` — even one that only surfaces while this CLI is building that
very failure's message, or, under `--verbose`, while it is building that
message's stack log — which propagates immediately, exactly as it would
from a command's own body, rather than becoming a JSON object at all.

```json
{"schema": "engrava.cli.error.v1", "error": "invalid_edge_type", "message": "Invalid edge type 'MADE_UP'; valid values: ASSOCIATED, DEPENDS_ON, DERIVED_FROM, MESSAGE_OF, BRIDGE, CONSOLIDATED_FROM, CONTESTED_BY"}
```

`schema` is a plain version string (not a URL) — there is no schema registry
to publish a URL against. `error` is a short machine-readable identifier
(`invalid_edge_type`, `missing_thought`, `database_not_found`,
`malformed_meta`, `malformed_filter`, `empty_text`, `invalid_top_k`,
`invalid_weight`, `invalid_config`, `unexpected_error`); `message` usually
contains the offending value, or — for `unexpected_error` — the underlying
exception's type and text. It does not always: an `invalid_config` or
`missing_thought` whose underlying `ConfigError` / `ReferentialIntegrityError`
is a third-party subclass, or whose relevant field (the config error's
`message`, or the referential error's `column` / `referenced_id`) is present
but not the plain type this CLI requires before using it, instead gets a
fixed message. That detail is **omitted**, not attempted and failed to
read: this CLI validates a field's type — and, for `missing_thought`, that
it names one of the two real columns and matches the id this invocation was
actually given — before ever reading it for display, and shows nothing at
all rather than a value it did not validate.

**Reading `--json` errors from stderr.** The JSON object above is always the
*last line* written to stderr, and nothing this CLI controls writes anything
else to stderr afterwards — the write happens once, after every cleanup a
failing invocation still had open (closing a database, for instance) has
already run to completion, never before it. It is not always the *only*
line, though: `remember` / `link` echo a `Created database: ...` notice to
stderr the first time they create the resolved database, before anything can
fail, and `--verbose` writes its own resolution notice (and, on an
unanticipated failure, the caught exception's stack — filename, line number
and function name per frame, not a full Python traceback, and typically
multiple lines) to stderr as the invocation proceeds — all of that precedes
the JSON object on a failing run, never follows it. Neither notice is
guaranteed to be a single line: both echo caller-supplied text (a path, an
exception's own message)
raw, and a value containing a literal line feed reproduces it, spreading
that one notice over more than one line of output.

A consumer should therefore `json.loads()` only the **last non-blank
element** of stderr, not the whole stream. `stderr.split("\n")` and
`str.splitlines()` both recover the same final element reliably: the error
object's own `ensure_ascii=True` encoding (see
`_emit_and_exit` in `engrava.cli.memory_commands`) guarantees it never
contains a literal U+0085/U+2028/U+2029 — the three Unicode line separators
`str.splitlines()` treats as breaks but a strict `"\n"` split does not — so
nothing inside the object itself can make the two recipes disagree about
where it ends; the test suite proves this directly by embedding a U+2028 in
a `--db` path and checking that both recipes decode the identical object.
What the two recipes do **not** agree on is the *earlier* lines: one of
those separators inside an echoed path or a stack line makes
`str.splitlines()` split that line into extra fragments, where a strict
`"\n"` split keeps it intact as one. That only matters to a consumer who
also parses the earlier lines — recovering an echoed `Created database:
...` path exactly, say — and for that, split strictly on `"\n"`. For
recovering only the final error object, either recipe is safe.

**What is never JSON, `--json` or not.** A failure Click itself rejects
*before* this command's own code — and therefore before `--json` has even
been parsed — always prints Click's own plain usage text to stderr and exits
`2`. This is **not a closed list**: it is everything Click's own argument
parser rejects on its way to calling this command's body, and that surface
belongs to Click, not to this CLI. Known examples include an invalid `--type`
/ `--priority` choice on `remember`, a `--top-k` or `--weight` that does not
parse as a number, a missing required argument (`TEXT`, `FROM`/`TO`, `link`'s
`--type`), an unknown option at the root or subcommand level, an unexpected
extra argument, an invalid root `--format`, and a missing value for any
option that takes one (`--db`, `--config`, `--meta`, `--filter`, `--top-k`,
`--weight`, `--type`, `--priority`). The rule for a consumer: **if `--json`'s
own JSON object was never confirmed to have been reached — i.e. you cannot
rule out a parse-phase rejection — do not assume stderr decodes**; attempt
`json.loads()` and fall back to treating the raw text as a Click usage error
on failure, rather than relying on an enumerated exception list (this one or
any other) to be complete.

One parse-phase footgun worth naming explicitly: an option that takes a
value (`--meta`, `--filter`, `--top-k`, `--weight`, `--type`, `--priority`)
**consumes the very next token as that value, even if it looks like another
flag**, so a missing value does not reliably produce an error at all —
Click's ordinary behaviour, not a bug, for these six. Use `--option=value`
(`=` syntax) when a value might otherwise be ambiguous with a following flag,
rather than relying on Click to catch the omission.

`--db` and `--config` are **not** on that list. A previous revision of this
document also called `engrava --db --json remember "x"` an instance of the
same ordinary behaviour — silently creating a database literally named
`--json` and never seeing the real `--json` flag at all — and left it alone.
That claim was false: the equivalent `argparse` program rejects the same
input outright ("expected one argument"), so calling it unavoidable parser
convention was wrong on the facts, not just generous. It is now a rejected
usage error instead (exit `2`, Click's own plain usage text, `--json` or
not — see **What is never JSON** above): any `--db` or `--config` value
starting with `-` is refused before it can be silently taken as a database
or config path. A caller who genuinely needs such a path can disambiguate it
the usual shell way, by prefixing it (`--db ./--json`).

### `remember`

Stores `TEXT` as a thought and prints its id. Built over `create_thought()`
with an explicitly constructed `ThoughtRecord` — **not** the library's
`remember()` shorthand, which takes only text, metadata, and a dedup flag and
always produces a `NOTE` / `P3` thought, so it cannot honour `--type` /
`--priority`.

| Option | Type | Default | Description |
|---|---|---|---|
| `TEXT` | positional | required | Content to store. `-` reads it from stdin instead. |
| `--type` | thought type | `NOTE` | One of `TASK`, `OBSERVATION`, `BELIEF`, `REFLECTION`, `OUTPUT_DRAFT`, `NOTE`. |
| `--priority` | priority | `P3` | One of `P1`, `P2`, `P3`, `P4` (`P1` highest). |
| `--meta` | `KEY=VALUE` | — | Metadata entry (repeatable). Values are stored as strings. |
| `--dedup` | flag | off | On identical existing content, bump `confirmation_count` and print the *existing* id instead of inserting a duplicate. |
| `--json` | flag | off | Emit a JSON object (schema `engrava.cli.remember.v1`) instead of a bare id. |

```bash
engrava --db my.db remember "User prefers concise answers"
engrava --db my.db remember "Escalate the outage" --type REFLECTION --priority P1
echo "piped content" | engrava --db my.db remember -
engrava --db my.db remember "tagged" --meta topic=weather --meta lang=en
engrava --db my.db remember "same content twice" --dedup   # run again: same id, no new row
```

`--json` output:

```json
{"schema": "engrava.cli.remember.v1", "thought_id": "091aa106-fcc0-45a3-a19b-d335ad05ea45", "deduplicated": false}
```

`deduplicated` is `true` only when `--dedup` was given **and** it matched an
existing thought — the printed `thought_id` is that existing thought's, not a
new row's.

### `recall`

Searches for thoughts relevant to `QUERY` and prints ranked results. Calls
the library `recall()` directly, so it behaves exactly like a direct library
call against the same database or config — unlike a bare connection, which
has no embedding provider and so no vector arm at all. This is not a
substitute for [`query`](#query): `query` runs structural MindQL (`FIND` /
`COUNT` / `SELECT`) with no ranking and no embedding provider of its own.

| Option | Type | Default | Description |
|---|---|---|---|
| `QUERY` | positional | required | Natural-language text to search for. |
| `--top-k` | int | `10` | Maximum results to return. |
| `--filter` | `KEY=VALUE` | — | Metadata equality filter (repeatable, AND-combined; flat keys only — nested-path filters are a later concern). |
| `--json` | flag | off | Emit a JSON object (schema `engrava.cli.recall.v1`) instead of a formatted table. |

```bash
engrava --db my.db recall "concise answers"
engrava --db my.db recall "tagged" --filter topic=weather --top-k 5
engrava --config engrava.yaml recall "concise answers" --json
```

A `KEY` missing an `=` is `malformed_filter`, exit `2` (see above). A `KEY`
present but outside the allowed key grammar (letters, digits, and
underscore only — no brackets, dots, spaces, `$`, or `..`) is not a
dedicated kind: it falls through to `unexpected_error`, exit `1`, since it is
the underlying filter library's own validation rejecting it, not a check
this command performs itself.

`--json` output:

```json
{"schema": "engrava.cli.recall.v1", "query": "concise answers", "top_k": 10, "backends_used": ["fts5", "priority", "vector"], "results": [{"thought_id": "091aa106-fcc0-45a3-a19b-d335ad05ea45", "score": 0.4714285714285715, "essence": "User prefers concise answers"}]}
```

`backends_used` names every search backend that was *available* for this
query — `"vector"` appears only when a configured embedding provider actually
reached the query (see [Store resolution](#store-resolution-remember--recall--link)
above); its absence with a `--config` you expect to configure one is the
degradation this command's whole design exists to avoid, not something to
silently tolerate.

### `link`

Creates a typed edge from `FROM` to `TO` and prints its id. Builds an
`EdgeRecord` and calls the public `create_edge()` — there is no public
`link()` to call instead.

| Option | Type | Default | Description |
|---|---|---|---|
| `FROM` | positional | required | Source thought id. |
| `TO` | positional | required | Target thought id. |
| `--type` | edge type | required | One of `ASSOCIATED`, `DEPENDS_ON`, `DERIVED_FROM`, `MESSAGE_OF`, `BRIDGE`, `CONSOLIDATED_FROM`, `CONTESTED_BY`. |
| `--weight` | float | `1.0` | Relation strength, `0.0`-`1.0`. |
| `--json` | flag | off | Emit a JSON object (schema `engrava.cli.link.v1`) instead of a bare id. |

```bash
engrava --db my.db link 091aa106-fcc0-45a3-a19b-d335ad05ea45 f4620859-3dfa-4f13-8d2e-d35df62dbba4 --type ASSOCIATED --weight 0.8
```

An unknown `--type` exits `2` naming the value and every valid member; a
`FROM` or `TO` that does not resolve to an existing thought exits `4`,
naming which one when available:

```bash
$ engrava --db my.db link ghost-id some-real-id --type ASSOCIATED
link: from_thought_id 'ghost-id' does not reference an existing thought.
$ echo $?
4
```

`--json` output:

```json
{"schema": "engrava.cli.link.v1", "edge_id": "b190dc41-9c87-4a66-9291-d70974fd2342", "from_thought_id": "091aa106-fcc0-45a3-a19b-d335ad05ea45", "to_thought_id": "f4620859-3dfa-4f13-8d2e-d35df62dbba4", "edge_type": "ASSOCIATED", "weight": 0.8}
```

### `snapshot`

Exports thoughts, edges, embeddings, and actions to a JSONL snapshot (one
record per line) — **not** the audit journal; see below.

| Option | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | path | derived (see below) | Output JSONL file path. |
| `--service` | name | see below | The service to snapshot (multi-service mode only). |

**Default output path** depends on the mode:

- **Single database:** `<db-stem>.snapshot.jsonl` next to the database — e.g.
  `--db engrava.db` → `engrava.snapshot.jsonl` (the `.db` suffix is replaced).
- **Multi-service:** `<data_dir>/<service>.snapshot.jsonl`.

**`--service`** resolves in three ways (see [Service resolution](#service-resolution)):

- **Explicit `--service NAME`** targets that service even with no services config
  — the service database is looked up in the data directory, which is the
  services config's `data_dir` if one is loaded, otherwise the **parent directory
  of `--db`**. `snapshot` only reads: if that service database does not already
  exist it prints `Service 'NAME' not found` and exits `1` without creating
  anything (`restore` is the command that creates a missing service database).
- **Omitted, with a services config loaded** → falls back to
  `services.default_service`.
- **Omitted, with no services config** → snapshots the single `--db` database.

```bash
engrava --db engrava.db snapshot -o backup.jsonl
engrava --db engrava.db snapshot               # -> engrava.snapshot.jsonl
engrava --db /data/engrava.db snapshot --service tenant_a   # -> /data/tenant_a.snapshot.jsonl
engrava --config engrava.yaml snapshot --service tenant_a   # data_dir from config
```

> A snapshot exports every column of the `thought`, `edge`, `embedding`, and
> `action` records — including the bi-temporal `valid_from` / `valid_until`
> fields — but **not** the audit journal (`journal_entry`). See
> [Backup & Recovery](backup-and-recovery.md) for what this means and when to use
> a physical file backup instead.

> **Failure safety.** `snapshot` writes to a temporary file next to `-o` and
> publishes it there only once every row has been read and the export's own
> read transaction has closed. On an exception or a cancelled run, an
> existing file at `-o` stays byte-identical rather than being replaced by a
> truncated one, and the temporary file is removed. A hard kill (`SIGKILL`)
> cannot run that cleanup, so `-o` then holds either the previous file or
> the complete new one, never a partial one, and a temporary file can be
> left behind and is safe to delete by hand. `-o` may not name the
> database currently open for `--db`, or that database's `-wal` / `-shm`
> companion files; pointing it there is refused
> before anything is written. An `-o` that is a symlink is followed to its
> real target, which is written and replaced, so the symlink itself is left
> pointing at the (now-updated) file; an `-o` that is a hard link instead
> becomes a new, separate file, so any other hard link to the old one keeps
> its old content.

### `restore`

Restores a database from a JSONL snapshot produced by `snapshot`.

| Option | Type | Default | Description |
|---|---|---|---|
| `-i`, `--input` | path | **required** | JSONL snapshot file to restore. |
| `--clear` | flag | off | Empty the target's four core tables and its journal before restoring (not `_metadata`, `extension_schema_versions`, or extension-owned tables). |
| `--clear-identity` | flag | off | Also clear the target's stored embedding identity (model name, dimension, prefix metadata). Requires `--clear`. Recovers a target whose stored `embedding_dimension` is corrupt -- `--clear` alone preserves an existing identity, corrupt or not. |
| `--skip-embeddings` | flag | off | Import without embedding records. |
| `--re-embed` | flag | off | Re-embed all thoughts via the target provider, ignoring source embeddings. Requires `--config` with top-level or per-service embeddings. |
| `--orphan-journal-entries` | flag | off | Allow a merge restore (no `--clear`) into a target whose `journal_entry` table is non-empty to replace a colliding row anyway. Without it, such a collision is refused. |
| `--service` | name | see below | The service to restore into. |

For any `restore --clear`, an existing sqlite-vec table is dropped in the same
transaction as the canonical rows. The next sqlite-vec-enabled open recreates
and backfills it. Because SQLite must load the virtual-table module before it can
remove that table safely, install `engrava[vec]` before clearing a database that
already contains a persisted sqlite-vec index.

`--service` resolves exactly as for [`snapshot`](#service-resolution): an explicit
`--service NAME` targets that service even without a services config (its database
resolves in the services `data_dir`, or the **parent directory of `--db`** when no
config is loaded); omitted with a services config falls back to
`services.default_service`; omitted with no services config restores into the
single `--db` database.

Thought and edge timestamps are written in the canonical UTC form engrava stores
(`2026-07-01T00:00:00+00:00`; a value without an offset is read as UTC), so a
snapshot taken by an earlier version, which may hold other ISO-8601 forms,
restores with its timestamps compared by instant. A value that cannot be read
as an ISO-8601 instant is restored as it was.

`--skip-embeddings` and `--re-embed` are **mutually exclusive** — passing both
fails with:

```
Error: --re-embed and --skip-embeddings are mutually exclusive.
```

Use `--re-embed` when the target should use a different embedding model than the
snapshot. In single-database mode, the provider comes from the top-level
`embeddings` section of the file passed with `--config`. In service mode,
`services.configs.<name>.embeddings` takes precedence; when no override exists,
the same top-level `embeddings` section is the fallback. Restore discards source
embedding rows, generates replacement vectors with the resolved provider, and
atomically replaces the stored model, dimension, document-prefix fingerprint,
and query-prefix pairing. A target that already contains embeddings requires
`--clear`; without it, restore refuses to relabel vectors that are not part of
the snapshot. The sqlite-vec reset described above prevents stale index rows
from surviving that replacement.

If neither level declares a provider, restore fails before importing records and
names the missing configuration. An explicit `--service` without a config-backed
provider fails for the same reason. Use `--skip-embeddings` to import text
without vectors, or restore normally to retain the snapshot vectors.

```bash
engrava --db fresh.db restore -i backup.jsonl
engrava --db fresh.db restore -i backup.jsonl --clear --skip-embeddings
engrava --db fresh.db --config engrava.yaml restore -i backup.jsonl --clear --re-embed
engrava --config engrava.yaml restore -i backup.jsonl --service main --clear --re-embed
```

When `services.default_service` names the configured target, `--service main`
may be omitted. Every restore — regardless of `--re-embed` or
`--skip-embeddings` — first checks that the target's *own* pre-existing
`embedding` rows agree with its *stored* embedding model, or with each other
when it has none: neither flag imports the snapshot's vectors as-is, so
neither can excuse a target that is already inconsistent going in. A normal
restore (neither `--re-embed` nor `--skip-embeddings`) additionally checks the
`model_name` and `dimension` every `embedding` row in the snapshot declares
against that same reference before inserting it — never against a configured
provider, which a plain restore never resolves, and never against the
snapshot's metadata header, which is not proof of anything the rows do not
already say for themselves. On a mismatch, restore fails before committing
anything, naming both identities; choose `--re-embed` to regenerate vectors for
the target's own model, or `--skip-embeddings` to import without vectors — an
already-inconsistent target itself is not fixed by either flag and needs its
own repair (or `--clear`) first.
Restoring embeddings into a target that starts with neither a stored model nor
any embeddings of its own writes the snapshot's declared identity as the
target's lock — the same `_metadata` write embedding directly into it would
make, though restore trusts the snapshot's declaration rather than computing
and measuring a vector itself.

**This check cannot see the document or query prefix the corpus was built
with** (see [Embeddings guide → Asymmetric prefixes for instruction-tuned
models](guides/embeddings.md#asymmetric-prefixes-for-instruction-tuned-models))
— a snapshot carries neither, only the vectors and a declared model name and
dimension for each. A target that already had a prefix
fingerprint locked before this restore keeps it unchanged (restore never
touches it), but a target that locks fresh from the snapshot's own vectors
(see above) records no prefix at all, regardless of what the source corpus
actually used. If the source corpus was built with a non-empty document
prefix, confirm that out of band before pointing a prefix-aware provider at
a restored target — restore itself cannot tell you.

> Restore recreates thoughts, edges, embeddings, and actions, **not** the audit
> journal. A **fresh target** therefore starts with an empty journal, and
> **`--clear`** empties the journal along with the data it wipes. A restore
> **without `--clear`** merges into the target — and if that target's journal
> is non-empty, **it refuses any record that collides** with an existing row
> (primary key or `UNIQUE` constraint) and rolls the whole restore back,
> rather than letting the merge silently replace it. Pass
> **`--orphan-journal-entries`** to allow the merge anyway; that restores the
> pre-gate behaviour, where a merge can orphan journal entries even when no
> incoming ID collides with one the journal describes — a duplicate
> `(from_thought_id, to_thought_id, edge_type)` triple replaces an existing
> edge, and replacing a thought cascades to that thought's own edges,
> embeddings, and actions — and `verify` still reports the chain as **valid**
> even though it no longer matches the data. The gate never applies to an
> empty journal, which is the ordinary case since journaling is opt-in.
> See [Backup & Recovery](backup-and-recovery.md#logical-snapshot-and-restore)
> for the full breakdown.

### `gc`

Garbage-collects `ARCHIVED` thoughts together with every edge touching one on
either end — including edges whose other end is still live — their embeddings and
the actions sourced from them, then reconciles the vector index by removing every
`vec0` row no `embedding` row owns. With `--expired` it also runs the TTL expiry
cleanup first.

> **`gc` now refuses on a database that is not on the current schema.** If
> `--db` (or the resolved service database) is below or above this build's
> head schema version, `gc` deletes nothing and exits `1` naming
> `engrava migrate` — see [Schema-version checks](#schema-version-checks)
> above. If a `gc` that used to run cleanly now exits non-zero, that is this
> refusal, not a regression in what it collects: run `engrava migrate` first.

| Option | Type | Default | Description |
|---|---|---|---|
| `--dry-run` | flag | off | Show what would be deleted without changing anything. |
| `--expired` | flag | off | Also run expiry cleanup (archive or delete per `ttl.strategy`) before collecting. |

```bash
engrava --db engrava.db gc                 # delete ARCHIVED thoughts + their edges/embeddings/actions
engrava --db engrava.db gc --expired       # run expiry cleanup first (per strategy)
engrava --db engrava.db gc --expired --dry-run
```

The behaviour of `gc --expired` depends on `ttl.strategy`: with `delete` it
removes expired rows and then collects pre-existing archived rows; with the
default `archive` it archives the expired rows and stops **only if it archived at
least one** — collecting the rows it just archived would defeat the soft-retire.
With no expired rows to archive it falls through to collecting pre-existing
`ARCHIVED` rows. See
[Data lifecycle → running cleanup](data-lifecycle.md#running-cleanup).

> **`gc` refuses to delete on a `vec0`-indexed store without the vector extra.**
> If the database carries an `embedding_vec` table and `sqlite-vec` cannot be
> loaded — most commonly because `engrava[vec]` is not installed, though an
> unsupported build, an OS error or a SQLite error fail the same way — a pass that
> is about to physically delete stops **before deleting anything** and exits `1`
> with:
>
> ```text
> Error: This database has a sqlite-vec index, and collecting thoughts without removing their vectors would strand them in it. Install 'engrava[vec]' and retry.
> ```
>
> Removing the rows without removing their vectors would strand those vectors in
> an index nothing can then reach them through. Only *deleting* passes are
> refused: `--dry-run` is never refused, and neither is a run with nothing to
> delete. Read the archive strategy carefully, though — `gc --expired` under the
> default `ttl.strategy: archive` stops after archiving **only when it actually
> archived something**. With no expired rows to archive it falls through to the
> archived-collection pass, which *is* refused when there are archived rows to
> collect. Install the extra (`pip install 'engrava[vec]'`) and retry.

### `migrate`

Runs pending schema migrations (ensures the core tables exist and are
up to date). Takes no command-specific options. Safe to run after an upgrade.

```bash
engrava --db engrava.db migrate
```

### `export`

Exports thoughts to a portable JSON file (with edges and metadata). Unlike
`snapshot` (JSONL, whole-database, for backup/restore), `export` writes a single
indented JSON document and can be filtered by lifecycle status.

| Option | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | path | `<db-stem>.export.json` (derived) | Output JSON file path. Written next to the database with the `.db` suffix replaced, e.g. `--db engrava.db` → `engrava.export.json`. |
| `--status` | lifecycle status | all | Only export thoughts with this `lifecycle_status` (e.g. `ACTIVE`). |

```bash
engrava --db engrava.db export -o thoughts.json
engrava --db engrava.db export --status ACTIVE
```

> **Failure safety.** Like `snapshot`, `export` writes to a temporary file
> next to `-o` and publishes it there only once every read has completed. On
> an exception or a cancelled run, an existing file at `-o` stays
> byte-identical and the temporary file is removed. A hard kill (`SIGKILL`)
> cannot run that cleanup, so `-o` then holds either the previous file or
> the complete new one, never a partial one, and a temporary file can be
> left behind and is safe to delete by hand. `-o` may not
> name the database currently open for `--db`, or that database's `-wal` /
> `-shm` companion files. A symlinked `-o` is followed and its real target
> replaced, leaving the link itself pointing at the new content; a
> hard-linked `-o` becomes a new, separate file instead.

## Journal verification

Use [`engrava verify`](#verify) to verify the [audit journal](audit-trail.md)'s
hash chain from the shell (exit `0` = intact, `1` = broken or missing database).
The equivalent Python API is `store.verify_journal()` (the store-level
convenience) or `store.journal.verify_integrity()` (via the writer directly):

```python
result = await store.verify_journal()
print(result.valid)
```

## See also

- [MindQL](mindql.md) — the query language `engrava query` runs
- [Backup & Recovery](backup-and-recovery.md) — snapshot/restore vs physical backup
- [Data Lifecycle](data-lifecycle.md) — what `gc` and `gc --expired` do
- [Configuration](configuration.md) — the `engrava.yaml` that `--config` loads
