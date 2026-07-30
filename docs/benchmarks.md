# Benchmarks

Engrava ships two reproducible benchmark harnesses:

- a small synthetic Dreaming benchmark used by the release smoke gate; and
- a LongMemEval runner for broader long-term-memory retrieval evaluation.

These are the two harnesses distributed **inside the `engrava` Python
package**. They are separate from the standalone
[`engrava-benchmark`](https://github.com/sovantica/engrava-benchmark)
repository, which provides its own cross-system reproduction workflow and
published result artifacts. Installing Engrava does not install or run that
repository, and the in-package commands below do not read its configurations or
results.

This page separates three things that are easy to conflate: the benchmark
**methodology**, the **release thresholds**, and the **observed values for one
specific Engrava and dependency version**. Passing the release thresholds does
not mean that Dreaming improves aggregate retrieval accuracy.

## Synthetic Dreaming benchmark

Install the local embedding extra and run the binding release measurements:

```bash
pip install 'engrava[embeddings-local]'
python -m engrava.benchmarks.synthetic
```

Add the frozen 50-question OFF/ON snapshot:

```bash
python -m engrava.benchmarks.synthetic --with-reproducibility
```

### Synthetic CLI options

| Option | Default | Scope |
|---|---:|---|
| `--with-reproducibility` | off | Also evaluate and print the bundled `synthetic-v1.json` OFF/ON per-scenario snapshot. The snapshot remains informational. |
| `--regenerate` | off | Generate a snapshot dataset instead of loading the bundled file. Implies `--with-reproducibility`; binding subsets are still fixed. |
| `--seed` | `20260508` | Generator seed used only with `--regenerate`. |
| `--n-conversations` | `50` | Generated conversation count; used only with `--regenerate`. |
| `--avg-turns` | `70` | Target turns per generated conversation; used only with `--regenerate`. |
| `--density` | `0.4` | Generated distraction density; used only with `--regenerate`. |
| `--scenarios` | all | Comma-separated known scenario names included in the reproducibility snapshot. Does not alter binding measurements. |
| `--top-k` | `5` | Retrieval K used by recall@K in the reproducibility snapshot and binding measurements that consume it. |
| `--output-format` | `text` | `text` or structured `json`. |
| `--output-path` | stdout | Used only with `--output-format json`; text output still goes to stdout. |

Exit code `0` means every binding measurement passed, `1` means at least one
failed, and `2` is the clean missing-embeddings-extras exit. Snapshot metrics do
not affect the exit code.

The harness uses `sentence-transformers/all-MiniLM-L6-v2`, a fixed generator
seed (`20260508`), deterministic scoring, and no LLM judge.

### Two measurement surfaces

The command deliberately reports two different surfaces.

**Binding release measurements** use separately generated, curated scenario
subsets. They gate the process exit code:

| Measurement | Contract | What it checks |
|---|---:|---|
| Synthesis coverage | `>= 0.80` | Dreaming creates REFLECTIONs whose membership intersects the expected synthesis facts. This is a data-layer mechanism check, not retrieval accuracy. |
| Direct neutrality | `<= 0.05` absolute ON/OFF delta | Dreaming does not materially move recall on the curated direct-answer subset. |
| Sanity tolerance | `<= 0.05` absolute ON/OFF delta | Dreaming does not materially move recall on the curated sanity subset. |
| Sanity, explicit boost `1.0` | `<= 0.05` absolute ON/OFF delta | The same sanity check with the reflection boost pinned explicitly. |

**The reproducibility snapshot** uses the bundled, frozen
`synthetic-v1.json`: 50 questions across nine naturally distributed scenarios.
It is informational and does not affect the exit code. This broader distribution
can expose regressions that the curated binding subsets do not.

## Observed v0.5/v0.6-candidate baseline

The v0.5.0 and v0.3.0 columns were measured on 2026-07-23; the v0.6.0 column was
**re-measured on 2026-07-30 against the release candidate itself**, because this
page's own reproduction rule forbids carrying a prior revision's numbers under a
release heading. Both measurements used:

- Python `3.12.13`;
- `sentence-transformers 5.6.0`;
- `sentence-transformers/all-MiniLM-L6-v2` (the harness default); and
- Engrava v0.5.0 revision `88b535b`.

The v0.6.0 column now reports the candidate at `2918e38`. The 2026-07-23 run had
measured a **provisional** candidate at `1033a2e`, twenty `src/` commits earlier —
including two in the dreaming path the binding measurements exercise — so those
numbers were superseded before the tag rather than wrong when taken.

**All four binding values are unchanged between the two candidate revisions.**
That is the substantive result of the re-measurement: those twenty commits moved
none of them. It is not a performance comparison — wall times across separate
runs are not comparable — and the byte-identity claim recorded for the earlier
pair is not restated here, because the reports were not compared byte for byte
this time.

### Binding results

| Measurement | v0.3.0 historical | v0.5.0 | v0.6.0 candidate | Contract |
|---|---:|---:|---:|---:|
| Synthesis coverage | `0.800` | `0.800` | `0.800` | `>= 0.80` |
| Direct neutrality | `0.033` | `0.000` | `0.000` | `<= 0.05` |
| Sanity tolerance | `0.042` | `0.000` | `0.000` | `<= 0.05` |
| Sanity, boost `1.0` | `0.042` | `0.000` | `0.000` | `<= 0.05` |

All binding measurements pass. These values come from the curated subsets, not
from the frozen 50-question distribution below.

### Frozen snapshot results

| Scenario | OFF recall@5 | ON recall@5 | Delta | OFF substring | ON substring |
|---|---:|---:|---:|---:|---:|
| **Overall (50 questions)** | **`0.800`** | **`0.700`** | **`-10.0 pp`** | **`0.660`** | **`0.600`** |
| `abstract_theme_recall` | `0.000` | `0.000` | `0.0 pp` | `0.000` | `0.000` |
| `contradiction_resolution` | `0.600` | `0.600` | `0.0 pp` | `0.200` | `0.200` |
| `distraction_heavy` | `1.000` | `1.000` | `0.0 pp` | `1.000` | `1.000` |
| `long_recall_simple` | `1.000` | `1.000` | `0.0 pp` | `1.000` | `1.000` |
| `multi_fact_recall` | `0.600` | `0.200` | `-40.0 pp` | `0.600` | `0.200` |
| `recent_fact_recall` | `0.900` | `0.900` | `0.0 pp` | `1.000` | `1.000` |
| `repeated_paraphrase_compression` | `1.000` | `1.000` | `0.0 pp` | `0.429` | `0.429` |
| `single_unique_fact` | `1.000` | `1.000` | `0.0 pp` | `1.000` | `1.000` |
| `thematic_cluster` | `0.833` | `0.333` | `-50.0 pp` | `0.833` | `0.667` |

The older v0.3.0 capture reported aggregate recall@5 `0.820` OFF and `0.780`
ON (`-4.0 pp`). Those numbers remain historical evidence for v0.3.0; they are
not a valid baseline for v0.5.0 or v0.6.0.

### Interpretation

- **No v0.5-to-v0.6 regression was observed.** Every recorded metric is
  identical between the two measured revisions.
- **Dreaming is not an accuracy-lift claim.** On the current frozen
  distribution, enabling it reduces aggregate recall@5 by 10 percentage points
  and substring match by 6 points. The largest recall movements are in
  `multi_fact_recall` and `thematic_cluster`.
- **A green binding gate and a weaker broad snapshot can coexist.** The binding
  measurements use different curated datasets and test bounded release
  contracts. They do not assert that the full frozen distribution improves.
- **Treat the snapshot as diagnostic.** It shows where Dreaming changes top-K
  outcomes; inspect result composition to determine whether REFLECTION
  participation caused a movement. Tune reflection participation for your
  corpus and measure your own workload.

### What `contradiction_resolution` does not prove

Despite its name, this scenario is a retrieval fixture, not a semantic conflict
engine. It stores an initial statement followed by a correction and asks for the
current state. Its recall metric accepts retrieval of either associated fact ID;
the separate substring metric checks whether the corrected value appears.

The v0.5/v0.6-candidate result is `0.600` recall and `0.200` corrected-value
substring match in both OFF and ON arms. It does not test entity resolution,
`CONTESTED_BY` edge creation, evidence promotion, truth selection, or
clarification-task generation. See [Evidence and conflicts](evidence-and-conflicts.md)
for Engrava's actual boundary.

## LongMemEval harness

Engrava also ships a runner for the public LongMemEval benchmark. The upstream
dataset is downloaded on first use and cached outside the package:

```bash
python -m engrava.benchmarks.longmemeval
python -m engrava.benchmarks.longmemeval --eval-mode=cosine --dreaming
```

### LongMemEval CLI options

| Option | Default | Scope |
|---|---:|---|
| `--variant` | `oracle` | Upstream dataset variant: `oracle`, `s`, or `m`. |
| `--eval-mode` | `substring` | CLI evaluator: `substring` or `cosine`. `llm` is Python-only. |
| `--subset` | all types | Exact `question_type` filter, for example `single-session-recall`. |
| `--limit` | no cap | Evaluate only the first N questions after the subset filter. |
| `--top-k` | `5` | Retrieval K passed to `search_hybrid()`. |
| `--dreaming` / `--no-dreaming` | off | Run, or explicitly suppress, one consolidation cycle per question. |
| `--hygiene` / `--no-hygiene` | off | Run, or suppress, one conservative archive-only Memory Hygiene pass per question after ingest/Dreaming. |
| `--hygiene-eviction-threshold` | `0.20` | Hygiene eviction-score cutoff; consulted only with `--hygiene`. Lower values forget less. |

The CLI prints a text summary and does not expose an output-file option. It
returns `0` after evaluating at least one question, `1` for download/runtime or
empty-filter failures, and `2` when the local embeddings extra is unavailable.
Each question uses a fresh in-memory database.

| Mode | LLM required | Role |
|---|---|---|
| `substring` | No | Deterministic Free-side reference mode. |
| `cosine` | No | Embedding-based match for paraphrased answers. |
| `llm` | Yes, caller-supplied judge | Optional programmatic evaluator; not exposed by the CLI. |

The default release smoke gate runs the synthetic binding measurements.
LongMemEval is an optional, longer manual probe because it downloads an external
dataset. Its harness, attribution, cache behavior, evaluation modes, and Python
API are documented in
[`src/engrava/benchmarks/longmemeval/README.md`](../src/engrava/benchmarks/longmemeval/README.md).
The Python API additionally exposes `db_path` for isolated per-question files
and `llm_judge` for caller-supplied LLM evaluation; neither is a CLI option.

## What these benchmarks do not establish

They do not prove:

- automatic contradiction detection or resolution;
- entity extraction, canonicalization, or truth arbitration;
- LLM answer quality (unless a caller explicitly supplies the LongMemEval LLM
  judge);
- production latency, throughput, or a corpus-size ceiling;
- tenant isolation, security, or audit-journal integrity; or
- that the synthetic scenario distribution matches your application.

Use the [Performance guide](performance.md) for workload measurement and the
[Security model](security.md) for trust boundaries.

## Reproducing and recording a release baseline

For a comparable rerun, hold all of these fixed:

- Engrava revision;
- benchmark dataset and generator seed;
- Python and numeric dependency versions;
- embedding model and provider version;
- search/Dreaming configuration; and
- `top_k`.

Record both the binding values and the full OFF/ON snapshot. A release may keep
the same thresholds while its observed values move; documentation must never
silently reuse a prior version's observed numbers.
