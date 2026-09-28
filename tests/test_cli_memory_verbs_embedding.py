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
Skipped when the package or its cached weights are not available -- e.g. a
minimal install without the ``embeddings-local`` extra, or CI's cache warm-up
having missed this model -- which an offline load reports as an ``OSError``
in more than one shape (see ``_model_available``). Anything that is not an
``OSError`` is a different class of failure and is left to surface as a real
test error.
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

from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402


def _model_available() -> bool:
    """Whether ``all-MiniLM-L12-v2`` is already cached locally.

    Measured directly, offline (``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``,
    an empty ``HF_HOME``, ``local_files_only=True``): a missing or incomplete
    local model cache is reported as a plain ``OSError``, but not always the
    same way. A wholly absent cache entry surfaces as an ``OSError`` caused by
    ``huggingface_hub``'s own ``LocalEntryNotFoundError`` (``transformers``
    catches that internally and re-raises the ``OSError`` ``from`` it). A
    cache that has the model's config but not its weights instead surfaces a
    **bare** ``OSError`` with no such cause at all -- the weight lookup in the
    installed ``transformers`` suppresses the underlying missing-entry error
    before raising. Both are "not cached"; an unrelated ``OSError`` raised
    during that same offline load (a genuine I/O failure, say) has no shape
    that reliably tells it apart from either, so it is skipped too rather than
    guessed at. Anything that is not an ``OSError`` at all is a different
    class of failure -- unrelated to cache state -- and propagates to fail
    the test.
    """
    try:
        sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2", local_files_only=True)
    except OSError:
        return False
    return True


def test_model_available_treats_a_missing_cache_entry_as_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wholly absent cache entry (``OSError`` caused by ``LocalEntryNotFoundError``) skips.

    Reproduces the shape ``transformers`` raises when the model is not in the
    local cache at all and ``local_files_only=True`` forbids a network call.
    This test does not need the model cached, so it is not marked with the
    class-level skip below.
    """

    class _UncachedSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            cause = LocalEntryNotFoundError("Cannot find the requested files in the disk cache")
            message = (
                "We couldn't connect to 'https://huggingface.co' to load the files, and "
                "couldn't find them in the cached files."
            )
            raise OSError(message) from cause

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _UncachedSentenceTransformer)
    assert _model_available() is False


def test_model_available_treats_a_bare_os_error_as_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache with the config but not the weights also skips, with no error cause at all.

    The installed ``transformers``' weight lookup suppresses the underlying
    missing-entry error and raises a bare ``OSError`` in that case -- there is
    no ``LocalEntryNotFoundError`` (or any other) cause to distinguish it from
    an unrelated I/O failure, so every ``OSError`` is treated the same way.
    This test does not need the model cached, so it is not marked with the
    class-level skip below.
    """

    class _WeightsMissingSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            message = "config cached, weights missing"
            raise OSError(message)

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _WeightsMissingSentenceTransformer
    )
    assert _model_available() is False


def test_model_available_lets_a_non_os_error_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that is not an ``OSError`` at all is a real failure, not a skip.

    An offline cache miss -- in any of its shapes -- is always reported as an
    ``OSError``; anything else (a ``RuntimeError``, say) is a different class
    of failure, unrelated to cache state, and must propagate and fail the
    test. This test does not need the model cached, so it is not marked with
    the class-level skip below.
    """

    class _BrokenSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            message = "incompatible package version"
            raise RuntimeError(message)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _BrokenSentenceTransformer)
    with pytest.raises(RuntimeError, match="incompatible package version"):
        _model_available()


def test_model_available_treats_a_permission_error_as_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``PermissionError`` is itself an ``OSError``, so it skips like any other.

    Documents the deliberate imprecision: a permission problem reading the
    cache cannot be told apart from a genuine cache miss by type or cause
    alone, so -- like every other ``OSError`` -- it is read as "not cached"
    rather than guessed at. This test does not need the model cached, so it
    is not marked with the class-level skip below.
    """

    class _DeniedSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            message = "cache directory not readable"
            raise PermissionError(message)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _DeniedSentenceTransformer)
    assert _model_available() is False


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


@pytest.mark.skipif(
    not _model_available(),
    reason="all-MiniLM-L12-v2 is not cached locally in this environment",
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
