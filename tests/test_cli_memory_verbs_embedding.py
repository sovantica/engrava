"""``recall --config`` honours a configured embedding provider.

This is the reason the shared store-resolution helper
(:mod:`engrava.cli.store_resolution`) exists at all: a bare connection has no
embedding provider, so copying that pattern for ``recall`` would silently
degrade every configured-embeddings deployment to lexical-only search. This
test asserts the **positive** — ``backends_used`` actually contains the
``"vector"`` arm — rather than only that a config-driven and a bare call
agree, since two silent lexical-only routes would agree with each other too.

Uses the real ``sentence-transformer`` provider, with the package's own
default model (``all-MiniLM-L12-v2``, see
``SentenceTransformerProvider``'s default) rather than a mock, because the
point is that the CLI reaches the *actual* configured provider through
``from_config()`` — a mock would only prove the CLI can call a mock. CI warms
this model into the HuggingFace cache alongside ``all-MiniLM-L6-v2`` before
this file's job runs (see ``warm-hf-cache`` in ``.github/workflows/ci.yml``).
Skipped when the package or its cached weights are not available, e.g. a
minimal install without the ``embeddings-local`` extra -- never for any other
failure, which is left to surface as a real test error.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from engrava.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


def _extract_json_object(output: str) -> dict:
    """Pull the one JSON object out of output that also carries a tqdm progress bar.

    Loading the sentence-transformer model writes a ``\\r``-updated progress
    bar with **no trailing newline or carriage return** before the CLI's own
    ``click.echo`` of the JSON payload runs — the two land concatenated on
    the same physical line, e.g. ``...5461.44it/s]{"schema": ...}``. Splitting
    on line boundaries does not isolate it; finding this command family's
    fixed ``{"schema"`` opening substring and parsing from there does.

    """
    marker = '{"schema"'
    idx = output.rfind(marker)
    if idx == -1:
        msg = f"no JSON object found in output: {output!r}"
        raise AssertionError(msg)
    return json.loads(output[idx:].strip())


sentence_transformers = pytest.importorskip(
    "sentence_transformers", reason="requires the 'embeddings-local' extra"
)


def _model_available() -> bool:
    """Whether ``all-MiniLM-L12-v2`` is already cached locally.

    ``local_files_only=True`` makes both ``sentence-transformers`` and the
    underlying ``huggingface_hub`` resolve entirely from the local cache and
    raise an ``OSError`` (directly, or a ``LocalEntryNotFoundError`` --
    itself an ``OSError`` subclass -- wrapped in one) when a required file is
    missing from it, never anything else. Catching only ``OSError`` keeps
    this a genuine "not cached locally" check: an unrelated bug in model
    loading (a bad config value, an incompatible package version) raises a
    different exception type and is left to fail the test instead of being
    absorbed into a silent skip.
    """
    try:
        sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2", local_files_only=True)
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason="all-MiniLM-L12-v2 is not cached locally in this environment",
)


def _write_embedding_config(config_path: Path, db_path: Path) -> None:
    config_path.write_text(
        "database:\n"
        f"  path: {db_path}\n"
        "embeddings:\n"
        "  provider: sentence-transformer\n"
        "  model: all-MiniLM-L12-v2\n"
        "  auto_embed: true\n",
        encoding="utf-8",
    )


class TestConfigDrivenEmbeddingArm:
    def test_recall_reports_vector_in_backends_used(self, tmp_path: Path) -> None:
        db = tmp_path / "vector.db"
        config_path = tmp_path / "engrava.yaml"
        _write_embedding_config(config_path, db)
        runner = CliRunner()

        remember_result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "remember",
                "The quick brown fox jumps over the lazy dog",
            ],
        )
        assert remember_result.exit_code == 0, remember_result.output

        recall_result = runner.invoke(
            cli, ["--config", str(config_path), "recall", "fox jumping", "--json"]
        )
        assert recall_result.exit_code == 0, recall_result.output
        payload = _extract_json_object(recall_result.output)
        assert "vector" in payload["backends_used"]

    def test_bare_recall_with_no_config_never_reports_vector(self, tmp_path: Path) -> None:
        """Control: the same query, with no ``--config`` at all, has no vector arm.

        Without this, a resolver that silently fell back to a bare store even
        when ``--config`` names a real provider would still pass the positive
        assertion above by coincidence if ``recall`` happened to report
        ``"vector"`` unconditionally — this pins the *other* side.
        """
        db = tmp_path / "bare.db"
        runner = CliRunner()
        runner.invoke(
            cli, ["--db", str(db), "remember", "The quick brown fox jumps over the lazy dog"]
        )

        recall_result = runner.invoke(cli, ["--db", str(db), "recall", "fox jumping", "--json"])
        assert recall_result.exit_code == 0, recall_result.output
        payload = _extract_json_object(recall_result.output)
        assert "vector" not in payload["backends_used"]
