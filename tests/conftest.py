"""Session-wide test environment hardening.

Imported by pytest before any test module, so the environment overrides
here take effect *before* ``torch`` / ``sentence-transformers`` are first
imported by a test. Two classes of flakiness are addressed:

* **Native thread oversubscription.** ``torch``, OpenMP and MKL each spin up
  a worker pool sized to the host CPU. Across a full suite run, many tests
  load a sentence-transformer model in-process; their pools contend for the
  same cores and, under enough accumulated pressure, a later model load can
  wedge (observed as a hang at low wall-clock progress with CPU far above
  wall time — classic thread thrashing). Pinning every native pool to a
  single thread removes the contention; the suite's model loads are one-shot
  encodes where extra threads buy nothing.
* **Tokenizer fork parallelism.** ``tokenizers`` warns about — and can
  deadlock on — parallelism across a fork. Disabling it keeps subprocess
  example/benchmark runs deterministic.

The offline flags are set **only when every in-process embedding model is
already cached**, so a warm developer/CI machine never reaches for the network
(the real cause of the intermittent full-suite hangs), while a cold-cache
environment that is *meant* to download is left untouched.

These overrides are deliberately conservative: every one of them only
removes nondeterminism (thread count, fork parallelism, network reach) and
none changes what the code under test computes.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Native thread pinning (must precede any torch / numpy import) ----------
# setdefault, so an explicit override from the caller's environment still wins.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --- Offline-when-cached for the in-process embedding models ----------------
#: Every sentence-transformer model the suite loads *in-process*. The synthetic
#: and LongMemEval benchmark surfaces resolve the default
#: (``engrava.benchmarks.synthetic.evaluate._DEFAULT_EMBEDDING_MODEL``), while
#: the quickstart example and its doc/example tests pin the L12 model. Offline
#: mode is forced only when *all* of these are already cached, so forcing it can
#: never starve a load the cache cannot satisfy.
_IN_PROCESS_EMBEDDING_MODELS = ("all-MiniLM-L6-v2", "all-MiniLM-L12-v2")


def _hf_hub_cache_dir() -> Path:
    """Return the Hugging Face Hub cache directory, honouring the env override."""
    override = os.environ.get("HF_HOME")
    if override:
        return Path(override) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_is_cached(model_name: str) -> bool:
    """Return ``True`` when ``model_name`` is present in the local Hub cache.

    Hub repos are stored as ``models--<org>--<name>`` (or ``models--<name>``
    for canonical repos). A substring match on the trailing model name is
    sufficient here: we only need to know whether forcing offline mode would
    starve a load that the cache can in fact satisfy.

    Args:
        model_name: The short model name, e.g. ``all-MiniLM-L6-v2``.

    Returns:
        ``True`` when a matching cached repo directory exists.
    """
    hub = _hf_hub_cache_dir()
    if not hub.is_dir():
        return False
    needle = model_name.replace("/", "--")
    return any(
        entry.name.startswith("models--") and needle in entry.name for entry in hub.iterdir()
    )


if all(_model_is_cached(name) for name in _IN_PROCESS_EMBEDDING_MODELS):
    # The cache can satisfy every in-suite model load, so forbid network
    # reach: this is what makes the full suite network-independent and
    # removes the intermittent socket-blocked hangs. A cold-cache run (any
    # model absent) is left free to download what it legitimately needs.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
