# Synthetic benchmark suite

Reproduce engrava's dreaming evidence on your machine in under 5 minutes.

## Quick start

```bash
pip install 'engrava[embeddings-local]'
python -m engrava.benchmarks.synthetic
```

Default invocation runs binding acceptance measurements (~5 minutes).
For full per-scenario texture:

```bash
python -m engrava.benchmarks.synthetic --with-reproducibility
```

## What it measures

Three properties of engrava's dreaming extension:

1. **Synthesis coverage** — does dreaming produce REFLECTIONs that
   consolidate related facts? (Data-layer mechanism check.)
2. **Direct retrieval neutrality** — does dreaming preserve baseline
   FTS/vector retrieval for questions with direct lexical answers?
3. **Sanity tolerance** — does dreaming avoid pathological influence
   on neutral queries?

Full interpretation guide + roadmap:
[`docs/benchmarks.md`](../../../../docs/benchmarks.md)

## Reproducibility commitment

Same seed + same bundled `synthetic-v1.json` + same engrava version
=> byte-identical results across runs.  The bundled dataset is frozen
for the v0.3.x line; future releases may add `synthetic-v2.json` etc.
without removing v1.
