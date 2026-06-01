# LongMemEval — Free harness

A reproducible engrava-side runner for the public **LongMemEval**
memory-evaluation benchmark. Use it to measure end-to-end retrieval
quality against the published dataset on any laptop, without API keys
in the default configuration.

## Attribution

LongMemEval was introduced by:

> Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu.
> *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
> Memory.*  ICLR 2025. arXiv:2410.10813.

The original dataset, evaluation protocol, and corpus are the work of
the LongMemEval authors. This package only wraps the dataset for
engrava-side ingestion + scoring.

## License + redistribution

- The upstream LongMemEval code repository is **MIT-licensed**
  (`github.com/xiaowu0162/LongMemEval`).
- The dataset is distributed via the upstream HuggingFace mirror under
  a permissive license.
- This package **does NOT bundle the dataset.** Engrava ships only the
  loader, runner, evaluator, and a small hand-authored test fixture
  (in `tests/benchmarks/fixtures/`) that mirrors the upstream schema
  but contains no upstream content.
- The loader downloads the requested variant at runtime to
  `~/.engrava/benchmarks/longmemeval/`. Cached files are reused on
  subsequent invocations.

If you need to operate offline, manually place the upstream JSON files
in that cache directory; the loader will skip the download.

## Download instructions

The loader handles this automatically on first invocation. If you
prefer to fetch the files manually:

```bash
mkdir -p ~/.engrava/benchmarks/longmemeval/
cd ~/.engrava/benchmarks/longmemeval/

# Oracle variant (smallest — recommended for smoke runs):
curl -L -o longmemeval_oracle.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

# S variant:
curl -L -o longmemeval_s_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# M variant (largest):
curl -L -o longmemeval_m_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
```

## Running the harness

The harness ships with engrava's embeddings extras. Install with:

```bash
pip install 'engrava[embeddings-local]'
```

Default invocation (oracle variant, substring mode, dreaming OFF):

```bash
python -m engrava.benchmarks.longmemeval
```

Cosine-mode against the oracle variant with dreaming ON:

```bash
python -m engrava.benchmarks.longmemeval --eval-mode=cosine --dreaming
```

Subset filter (one question type only):

```bash
python -m engrava.benchmarks.longmemeval --subset=single-session-recall
```

## Evaluation modes

| Mode | Deterministic | LLM required | When to use |
|---|---|---|---|
| `substring` (default) | yes | no | Reference for the Free-side quality gate. |
| `cosine` | yes (given fixed provider) | no | Catches paraphrased matches the substring check misses. |
| `llm` | depends on judge | yes (user-supplied) | Closest to the canonical LongMemEval scoring; opt-in. |

The CLI exposes `substring` and `cosine`. The `llm` mode is
deliberately **not** wired through the CLI — argparse rejects
`--eval-mode=llm`. To use the LLM judge, drive `run_longmemeval`
directly from Python and pass an `llm_judge` argument that implements
the `engrava.benchmarks.longmemeval.evaluate.LLMJudgeClient` protocol.
SDK wiring varies per provider and engrava does not pin one — see the
snippet below.

### Storage semantics

By default every question runs in its own `:memory:` SQLite database
(LongMemEval forbids cross-question state). Passing `db_path=<dir>` to
`run_longmemeval` switches to on-disk storage: the runner creates one
DB file per `question_id` inside that directory, so on-disk runs also
keep the haystacks fully isolated.

## Programmatic usage

Default (substring mode, in-memory store per question):

```python
import asyncio

from engrava.benchmarks.longmemeval import load_dataset
from engrava.benchmarks.longmemeval.runner import run_longmemeval
from engrava.benchmarks.synthetic.evaluate import (
    resolve_embedding_provider_or_exit,
)


async def main() -> None:
    questions = load_dataset(variant="oracle")
    provider = resolve_embedding_provider_or_exit()
    result = await run_longmemeval(
        questions,
        dreaming_enabled=False,
        embedding_provider=provider,
        eval_mode="substring",
    )
    print(f"aggregate score: {result.aggregate_score:.4f}")


asyncio.run(main())
```

LLM-judge mode (caller supplies any client matching `LLMJudgeClient`):

```python
import asyncio

from engrava.benchmarks.longmemeval import load_dataset
from engrava.benchmarks.longmemeval.runner import run_longmemeval
from engrava.benchmarks.synthetic.evaluate import (
    resolve_embedding_provider_or_exit,
)


class MyJudge:
    """Thin wrapper around any LLM SDK (Anthropic, OpenAI, local)."""

    async def judge(self, *, system: str, user: str) -> str:
        # Call your SDK here; return the model's reply.
        ...


async def main() -> None:
    questions = load_dataset(variant="oracle")
    provider = resolve_embedding_provider_or_exit()
    result = await run_longmemeval(
        questions,
        dreaming_enabled=False,
        embedding_provider=provider,
        eval_mode="llm",
        llm_judge=MyJudge(),
    )
    print(f"aggregate score: {result.aggregate_score:.4f}")


asyncio.run(main())
```

## Determinism

Substring mode is byte-deterministic across runs given the same dataset
and the same engrava configuration. Cosine mode is deterministic given
a fixed embedding provider. LLM mode is non-deterministic by design
(provider sampling), so its results carry an informational role.
