"""Tests for service isolation and export/import.

Covers:
- ServicesConfig / ServiceConfig value objects
- _parse_services YAML parser
- EngravaManager (multi-service isolation, lazy init, close_all)
- Export/import JSONL with --re-embed, --skip-embeddings, --service
- Backward compatibility with single-service mode
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
from pathlib import Path

import aiosqlite
import pytest
from click.testing import CliRunner

from engrava import (
    ConfigError,
    EdgeRecord,
    EdgeType,
    EmbeddingConfig,
    EngravaConfig,
    HygienePolicyConfig,
    LifecycleStatus,
    MetricsConfig,
    Priority,
    SearchConfig,
    ServiceConfig,
    ServicesConfig,
    SqliteEngravaCore,
    ThoughtRecord,
    ThoughtType,
)
from engrava.cli.main import cli
from engrava.config import (
    _parse_services,
    _validate_service_name,
    load_config,
)
from engrava.domain.protocols.derived_records import DeriveGates
from engrava.infrastructure.service_manager import EngravaManager


def _entry_names(directory: Path) -> list[str]:
    """Sorted names directly under *directory* (kept out of the async test body)."""
    return sorted(entry.name for entry in directory.iterdir())


class _LyingName(str):
    """A service name whose text and whose rendering disagree.

    Every character-level check — the name pattern above all — reads the real
    text ``"prod"`` and passes. ``__format__`` then answers with a relative path
    that climbs out of the data directory, so any ``f"{name}.db"`` built from
    this object addresses a file the store was never given.
    """

    __slots__ = ()

    def __format__(self, format_spec: str) -> str:
        del format_spec
        return "../escaped"


class _LyingCacheKey(str):
    """A service name that refuses to equal itself.

    ``__eq__`` and ``__hash__`` decide which entry of the store cache a name
    resolves to. A name that answers them for itself resolves to no entry, so
    the same service opens twice over one database file.
    """

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        del other
        return False

    def __ne__(self, other: object) -> bool:
        del other
        return True

    def __hash__(self) -> int:
        return 0


class _HostileRepr:
    """A rejected value whose ``repr`` raises."""

    def __repr__(self) -> str:
        msg = "repr ran caller-controlled code"
        raise RuntimeError(msg)


class _HostileNameMeta(type):
    """Metaclass whose ``__name__`` raises, like a hostile third-party type."""

    @property
    def __name__(cls) -> str:
        msg = "type name lookup ran caller-controlled code"
        raise RuntimeError(msg)


class _HostileTypeName(metaclass=_HostileNameMeta):
    """A rejected value whose *type name* raises.

    ``repr`` is deliberately well-behaved: naming the type is the other half of
    the "describe what you rejected" reflex, and it is a caller-controlled
    lookup just as much as ``repr`` is.
    """

    def __repr__(self) -> str:
        return "<a value whose type name raises>"


class _RestoreEmbeddingProvider:
    """Deterministic provider used by CLI restore configuration tests."""

    dimension = 3

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    async def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(self.query_prefix + text)

    async def embed_document(self, text: str) -> list[float]:
        return await self.embed(self.document_prefix + text)

    async def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch([self.query_prefix + text for text in texts])

    async def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_batch([self.document_prefix + text for text in texts])


# ------------------------------------------------------------------
# Value objects: ServiceConfig, ServicesConfig
# ------------------------------------------------------------------


class TestServiceConfig:
    def test_defaults(self) -> None:
        cfg = ServiceConfig()
        assert cfg.embeddings is None

    def test_with_embeddings(self) -> None:
        emb = EmbeddingConfig(provider="ollama", model="nomic-embed-text")
        cfg = ServiceConfig(embeddings=emb)
        assert cfg.embeddings is not None
        assert cfg.embeddings.provider == "ollama"

    def test_frozen(self) -> None:
        cfg = ServiceConfig()
        with pytest.raises(AttributeError):
            cfg.embeddings = None  # type: ignore[misc]


class TestServicesConfig:
    def test_defaults(self) -> None:
        cfg = ServicesConfig(data_dir=Path("./data"))
        assert cfg.default_service == "main"
        assert cfg.configs == {}

    def test_with_configs(self) -> None:
        cfg = ServicesConfig(
            data_dir=Path("./data"),
            default_service="alpha",
            configs={"alpha": ServiceConfig()},
        )
        assert cfg.default_service == "alpha"
        assert "alpha" in cfg.configs

    def test_frozen(self) -> None:
        cfg = ServicesConfig(data_dir=Path("./data"))
        with pytest.raises(AttributeError):
            cfg.data_dir = Path("./other")  # type: ignore[misc]


# ------------------------------------------------------------------
# Service name validation
# ------------------------------------------------------------------


class TestServiceNameValidation:
    def test_valid_names(self) -> None:
        for name in ("main", "alpha", "home-assistant", "coding_helper", "a1b2"):
            assert _validate_service_name(name) == name

    def test_returns_an_exact_str_for_a_subclass(self) -> None:
        """The name handed back is one the module owns, not the caller's object.

        A ``str`` subclass passes the pattern check (the pattern reads the real
        text) and then answers ``__format__`` however it likes. The validator
        exists to hand callers a value with no such freedom.
        """
        validated = _validate_service_name(_LyingName("prod"))
        assert type(validated) is str
        assert validated == "prod"
        assert f"{validated}.db" == "prod.db"

    def test_rejects_a_non_string_with_a_config_error(self) -> None:
        """A non-string is a domain rejection, not a leaked ``TypeError``."""
        with pytest.raises(ConfigError, match="must be a string"):
            _validate_service_name(object())

    def test_rejection_message_never_calls_the_caller_repr(self) -> None:
        """Building the refusal must not call back into caller-controlled code.

        ``repr`` is a lookup on something the caller supplies and can raise from
        inside the rejection path, turning a clean refusal into an unrelated
        error. (The adversaries here are built in the test body rather than
        parametrised so that a failure stays reportable — an object that fights
        ``repr`` also fights the test runner printing it.)
        """
        with pytest.raises(ConfigError, match="must be a string"):
            _validate_service_name(_HostileRepr())

    def test_rejection_message_never_reads_the_caller_type_name(self) -> None:
        """The other half of the same reflex: naming the rejected value's type."""
        with pytest.raises(ConfigError, match="must be a string"):
            _validate_service_name(_HostileTypeName())

    def test_invalid_uppercase(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _validate_service_name("MyService")

    def test_invalid_start_digit(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _validate_service_name("1service")

    def test_invalid_spaces(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _validate_service_name("my service")

    def test_invalid_empty(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _validate_service_name("")

    def test_invalid_too_long(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _validate_service_name("a" * 64)

    def test_max_length_ok(self) -> None:
        _validate_service_name("a" * 63)  # should not raise


# ------------------------------------------------------------------
# _parse_services YAML parser
# ------------------------------------------------------------------


class TestParseServices:
    def test_none_returns_none(self) -> None:
        assert _parse_services(None) is None

    def test_minimal(self) -> None:
        raw = {"data_dir": "./data/engrava"}
        cfg = _parse_services(raw)
        assert cfg is not None
        assert cfg.data_dir == Path("./data/engrava")
        assert cfg.default_service == "main"
        assert cfg.configs == {}

    def test_full(self) -> None:
        raw = {
            "data_dir": "./data/engrava",
            "default_service": "alpha",
            "configs": {
                "alpha": {
                    "embeddings": {
                        "provider": "sentence-transformer",
                        "model": "all-MiniLM-L12-v2",
                    },
                },
                "helper": None,
            },
        }
        cfg = _parse_services(raw)
        assert cfg is not None
        assert cfg.default_service == "alpha"
        assert "alpha" in cfg.configs
        assert cfg.configs["alpha"].embeddings is not None
        assert cfg.configs["alpha"].embeddings.provider == "sentence-transformer"
        assert "helper" in cfg.configs
        assert cfg.configs["helper"].embeddings is None

    def test_missing_data_dir(self) -> None:
        with pytest.raises(ConfigError, match="data_dir"):
            _parse_services({"default_service": "main"})

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            _parse_services("invalid")

    def test_invalid_service_name_in_configs(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _parse_services(
                {
                    "data_dir": "./data",
                    "configs": {"INVALID": {}},
                }
            )

    def test_invalid_default_service(self) -> None:
        with pytest.raises(ConfigError, match="Invalid service name"):
            _parse_services(
                {
                    "data_dir": "./data",
                    "default_service": "WRONG",
                }
            )


class TestLoadConfigWithServices:
    def test_single_service_mode(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n  path: ./test.db\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.services is None

    def test_multi_service_mode(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            "database:\n  path: ./test.db\n"
            "services:\n"
            "  data_dir: ./data/engrava\n"
            "  default_service: alpha\n"
            "  configs:\n"
            "    alpha:\n"
            "      embeddings:\n"
            "        provider: ollama\n"
            "        model: nomic-embed-text\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.services is not None
        assert config.services.default_service == "alpha"
        assert "alpha" in config.services.configs


# ------------------------------------------------------------------
# EngravaManager
# ------------------------------------------------------------------


class TestEngravaManager:
    async def test_get_store_creates_db(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            store = await mgr.get_store("alpha")
            assert store is not None
            db_file = data_dir / "alpha.db"
            assert db_file.exists()

    async def test_get_store_caches(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            store1 = await mgr.get_store("alpha")
            store2 = await mgr.get_store("alpha")
            assert store1 is store2

    async def test_multiple_services_isolated(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            store_a = await mgr.get_store("svc-a")
            store_b = await mgr.get_store("svc-b")

            thought = ThoughtRecord(
                thought_id="t-001",
                essence="Alpha thought",
                content="Belongs to svc-a only",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
            await store_a.create_thought(thought)

            # svc-b should not see svc-a's thought.
            result = await store_b.get_thought("t-001")
            assert result is None

    async def test_list_services(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            assert await mgr.list_services() == []
            await mgr.get_store("beta")
            await mgr.get_store("alpha")
            services = await mgr.list_services()
            assert services == ["alpha", "beta"]

    async def test_delete_service(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            await mgr.get_store("doomed")
            assert (data_dir / "doomed.db").exists()
            await mgr.delete_service("doomed")
            assert not (data_dir / "doomed.db").exists()
            assert await mgr.list_services() == []

    async def test_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        data_dir.mkdir()
        async with EngravaManager(data_dir=data_dir) as mgr:
            with pytest.raises(FileNotFoundError):
                await mgr.delete_service("ghost")

    async def test_close_all(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        mgr = EngravaManager(data_dir=data_dir)
        await mgr.get_store("one")
        await mgr.get_store("two")
        await mgr.close_all()
        # Stores cleared after close_all.
        assert len(mgr._stores) == 0

    async def test_context_manager(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            await mgr.get_store("ctx-svc")
        # After __aexit__, stores should be cleared.
        assert len(mgr._stores) == 0

    async def test_invalid_service_name(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            with pytest.raises(ConfigError, match="Invalid service name"):
                await mgr.get_store("INVALID")

    async def test_per_service_schema_independent(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            store_a = await mgr.get_store("svc-a")
            store_b = await mgr.get_store("svc-b")

            # Both have independent schema versions.
            cursor_a = await store_a._db.execute("PRAGMA user_version")
            row_a = await cursor_a.fetchone()
            cursor_b = await store_b._db.execute("PRAGMA user_version")
            row_b = await cursor_b.fetchone()
            assert row_a[0] == row_b[0] == 20

    async def test_per_service_fts_independent(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            store_a = await mgr.get_store("svc-a")
            store_b = await mgr.get_store("svc-b")

            # Insert into svc-a's FTS.
            thought = ThoughtRecord(
                thought_id="t-fts",
                essence="FTS isolation test",
                content="Unique content for FTS",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
            await store_a.create_thought(thought)

            # svc-b FTS should be empty.
            cursor = await store_b._db.execute(
                "SELECT COUNT(*) FROM thought_fts WHERE thought_fts MATCH 'isolation'"
            )
            row = await cursor.fetchone()
            assert row[0] == 0

    async def test_per_service_embedding_isolated(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        async with EngravaManager(data_dir=data_dir) as mgr:
            store_a = await mgr.get_store("svc-a")
            store_b = await mgr.get_store("svc-b")

            thought = ThoughtRecord(
                thought_id="t-emb",
                essence="Embedding isolation",
                content="Test content",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=1,
                updated_cycle=1,
            )
            await store_a.create_thought(thought)
            await store_a.store_embedding("t-emb", [1.0] * 16)

            # svc-b embedding table should be empty.
            cursor = await store_b._db.execute("SELECT COUNT(*) FROM embedding")
            row = await cursor.fetchone()
            assert row[0] == 0

    async def test_from_config(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        svc_cfg = ServicesConfig(
            data_dir=data_dir,
            default_service="main",
        )
        mgr = await EngravaManager.from_config(svc_cfg)
        try:
            store = await mgr.get_store("main")
            assert store is not None
        finally:
            await mgr.close_all()

    async def test_per_service_embedding_config(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "services"
        svc_cfg = ServicesConfig(
            data_dir=data_dir,
            configs={
                "custom": ServiceConfig(
                    embeddings=EmbeddingConfig(provider=None),
                ),
            },
        )
        mgr = EngravaManager(
            data_dir=data_dir,
            default_embeddings=EmbeddingConfig(provider=None),
            services_config=svc_cfg,
        )
        try:
            store = await mgr.get_store("custom")
            assert store is not None
        finally:
            await mgr.close_all()


# ------------------------------------------------------------------
# Export/Import JSONL (snapshot/restore with new flags)
# ------------------------------------------------------------------


@pytest.fixture
def ms11_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def ms11_populated_db(tmp_path: Path) -> Path:
    """Create a DB with sample data for export/import tests."""
    db_path = tmp_path / "test.db"

    async def _setup() -> None:
        conn = await aiosqlite.connect(str(db_path))
        conn.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(conn)
        await store.ensure_schema()

        for i in range(2):
            t = ThoughtRecord(
                thought_id=f"t-{i:03d}",
                essence=f"Thought {i}",
                content=f"Content number {i}",
                thought_type=ThoughtType.OBSERVATION,
                source="test",
                lifecycle_status=LifecycleStatus.ACTIVE,
                priority=Priority.P2,
                created_cycle=i,
                updated_cycle=i,
            )
            await store.create_thought(t)
            await store.store_embedding(f"t-{i:03d}", [float(i)] * 16)

        edge = EdgeRecord(
            edge_id="e-001",
            from_thought_id="t-000",
            to_thought_id="t-001",
            edge_type=EdgeType.ASSOCIATED,
            weight=0.8,
            created_cycle=1,
        )
        await store.create_edge(edge)

        # Lock embedding model in _metadata.
        await conn.execute(
            "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
            ("embedding_model_name", "all-MiniLM-L12-v2"),
        )
        await conn.execute(
            "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
            ("embedding_dimension", "16"),
        )
        await conn.commit()
        await conn.close()

    asyncio.run(_setup())
    return db_path


class TestSnapshotNewFormat:
    def test_snapshot_includes_metadata_header(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "snap.jsonl"
        result = ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(out)],
        )
        assert result.exit_code == 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        header = json.loads(lines[0])
        assert header["_type"] == "metadata"
        assert header["schema_version"] == 20
        assert header["embedding_model_name"] == "all-MiniLM-L12-v2"
        assert header["embedding_dimension"] == 16

    def test_snapshot_data_records_use_type_field(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "snap.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(out)],
        )
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        # Skip metadata header.
        for line in lines[1:]:
            record = json.loads(line)
            assert "_type" in record
            assert record["_type"] in {"thought", "edge", "embedding", "action"}
            assert "data" in record


class TestRestoreWithFlags:
    def test_restore_skip_embeddings(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "snap.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        new_db = tmp_path / "restored.db"
        result = ms11_runner.invoke(
            cli,
            ["--db", str(new_db), "restore", "-i", str(snap), "--skip-embeddings"],
        )
        assert result.exit_code == 0

        # Verify thoughts imported but embeddings skipped.
        async def _check() -> tuple[int, int]:
            conn = await aiosqlite.connect(str(new_db))
            conn.row_factory = aiosqlite.Row
            tc = await (await conn.execute("SELECT COUNT(*) FROM thought")).fetchone()
            ec = await (await conn.execute("SELECT COUNT(*) FROM embedding")).fetchone()
            await conn.close()
            return tc[0], ec[0]

        thought_count, embed_count = asyncio.run(_check())
        assert thought_count == 2
        assert embed_count == 0

    def test_restore_re_embed_and_skip_mutually_exclusive(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "dummy.jsonl"
        snap.write_text("", encoding="utf-8")
        db = tmp_path / "test.db"
        result = ms11_runner.invoke(
            cli,
            ["--db", str(db), "restore", "-i", str(snap), "--re-embed", "--skip-embeddings"],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_restore_backward_compat_legacy_format(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Ensure old-format snapshots ({"table": ...}) still import."""
        snap = tmp_path / "legacy.jsonl"
        snap.write_text(
            json.dumps(
                {
                    "table": "thought",
                    "data": {
                        "thought_id": "legacy-001",
                        "thought_type": "OBSERVATION",
                        "essence": "Legacy",
                        "content": "Legacy content",
                        "priority": "P2",
                        "lifecycle_status": "ACTIVE",
                        "source": "test",
                        "created_cycle": 1,
                        "updated_cycle": 1,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        new_db = tmp_path / "legacy-restore.db"
        result = ms11_runner.invoke(
            cli,
            ["--db", str(new_db), "restore", "-i", str(snap)],
        )
        assert result.exit_code == 0

        async def _check() -> int:
            conn = await aiosqlite.connect(str(new_db))
            conn.row_factory = aiosqlite.Row
            tc = await (await conn.execute("SELECT COUNT(*) FROM thought")).fetchone()
            await conn.close()
            return tc[0]

        assert asyncio.run(_check()) == 1

    def test_snapshot_restore_roundtrip_new_format(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "snap.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        new_db = tmp_path / "roundtrip.db"
        result = ms11_runner.invoke(
            cli,
            ["--db", str(new_db), "restore", "-i", str(snap)],
        )
        assert result.exit_code == 0

        check = ms11_runner.invoke(
            cli,
            ["--db", str(new_db), "--format", "json", "info"],
        )
        data = json.loads(check.output)
        assert data["thoughts"]["total"] == 2
        assert data["edges"]["total"] == 1


class TestServiceSnapshot:
    def test_snapshot_with_service(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "services"

        async def _setup() -> None:
            async with EngravaManager(data_dir=data_dir) as mgr:
                store = await mgr.get_store("alpha")
                t = ThoughtRecord(
                    thought_id="svc-t-001",
                    essence="Service thought",
                    content="Alpha content",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=1,
                    updated_cycle=1,
                )
                await store.create_thought(t)

        asyncio.run(_setup())

        out = tmp_path / "alpha.jsonl"
        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(data_dir / "any.db"),
                "snapshot",
                "--service",
                "alpha",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        # At least metadata + 1 thought.
        assert len(lines) >= 2

    def test_restore_to_service(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        src_dir = tmp_path / "src-services"
        dst_dir = tmp_path / "dst-services"

        # Create source with data.
        async def _setup() -> None:
            async with EngravaManager(data_dir=src_dir) as mgr:
                store = await mgr.get_store("source")
                t = ThoughtRecord(
                    thought_id="cross-t-001",
                    essence="Cross-service",
                    content="Content",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=1,
                    updated_cycle=1,
                )
                await store.create_thought(t)

        asyncio.run(_setup())

        # Export from source.
        snap = tmp_path / "source.jsonl"
        ms11_runner.invoke(
            cli,
            [
                "--db",
                str(src_dir / "any.db"),
                "snapshot",
                "--service",
                "source",
                "-o",
                str(snap),
            ],
        )

        # Import into target.
        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(dst_dir / "any.db"),
                "restore",
                "-i",
                str(snap),
                "--service",
                "target",
            ],
        )
        assert result.exit_code == 0

        # Verify target has the thought.
        async def _check() -> int:
            async with EngravaManager(data_dir=dst_dir) as mgr:
                store = await mgr.get_store("target")
                cursor = await store._db.execute("SELECT COUNT(*) FROM thought")
                row = await cursor.fetchone()
                return row[0]

        assert asyncio.run(_check()) == 1


class TestEngravaConfigServices:
    """Verify EngravaConfig includes services field."""

    def test_default_none(self) -> None:
        cfg = EngravaConfig(database_path=Path("./test.db"))
        assert cfg.services is None

    def test_with_services(self) -> None:
        svc = ServicesConfig(data_dir=Path("./data"))
        cfg = EngravaConfig(database_path=Path("./test.db"), services=svc)
        assert cfg.services is not None
        assert cfg.services.data_dir == Path("./data")


# ------------------------------------------------------------------
# Regression tests for quality-gap fixes
# ------------------------------------------------------------------


class TestReEmbedRequiresProvider:
    """Fix 1: --re-embed must fail when no embedding provider is configured."""

    def test_re_embed_uses_top_level_provider_for_single_db(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        snap = tmp_path / "direct-source.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        target_db = tmp_path / "direct-target.db"
        config_path = tmp_path / "direct.yaml"
        config_path.write_text(
            f"database:\n  path: {target_db}\n"
            "embeddings:\n"
            "  provider: ollama\n"
            "  model: direct-model\n"
            '  query_prefix: "query: "\n'
            '  document_prefix: "passage: "\n',
            encoding="utf-8",
        )

        async def _seed_stale_metadata() -> None:
            conn = await aiosqlite.connect(str(target_db))
            store = SqliteEngravaCore(conn)
            await store.ensure_schema()
            await conn.executemany(
                "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                [
                    ("embedding_model_name", "stale-model"),
                    ("embedding_dimension", "999"),
                    ("embedding_document_prefix_fingerprint", "stale-document-prefix"),
                    ("embedding_query_prefix", "stale-query-prefix"),
                ],
            )
            await conn.execute(
                "CREATE TABLE embedding_vec(rowid INTEGER PRIMARY KEY, embedding BLOB)"
            )
            await conn.execute(
                "INSERT INTO embedding_vec(rowid, embedding) VALUES (1, X'00000000')"
            )
            await conn.commit()
            await conn.close()

        asyncio.run(_seed_stale_metadata())
        resolved_providers: list[str | None] = []

        def _resolve(config: EmbeddingConfig | None) -> _RestoreEmbeddingProvider | None:
            resolved_providers.append(config.provider if config else None)
            if config is None or config.provider is None:
                return None
            return _RestoreEmbeddingProvider(
                config.model or config.provider,
                query_prefix=config.query_prefix or "",
                document_prefix=config.document_prefix or "",
            )

        monkeypatch.setattr("engrava.cli.main.resolve_embedding_provider", _resolve)

        async def _load_existing_vec_index(_conn: aiosqlite.Connection) -> bool:
            return True

        monkeypatch.setattr(
            "engrava.extensions.vector_sqlite_vec.load_sqlite_vec",
            _load_existing_vec_index,
        )

        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(target_db),
                "--config",
                str(config_path),
                "restore",
                "-i",
                str(snap),
                "--re-embed",
            ],
        )

        assert result.exit_code == 0
        assert resolved_providers == ["ollama"]

        async def _read_corpus_identity() -> tuple[dict[str, str], set[str], bool]:
            conn = await aiosqlite.connect(str(target_db))
            metadata_cursor = await conn.execute(
                "SELECT key, value FROM _metadata WHERE key LIKE 'embedding_%'"
            )
            metadata = {str(key): str(value) for key, value in await metadata_cursor.fetchall()}
            owner_cursor = await conn.execute("SELECT DISTINCT owner_type FROM embedding")
            owner_types = {str(row[0]) for row in await owner_cursor.fetchall()}
            vec_cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_vec'"
            )
            vec_index_exists = await vec_cursor.fetchone() is not None
            await conn.close()
            return metadata, owner_types, vec_index_exists

        metadata, owner_types, vec_index_exists = asyncio.run(_read_corpus_identity())
        expected_fingerprint = hashlib.sha256(b"passage: ").hexdigest()
        assert metadata == {
            "embedding_model_name": "direct-model",
            "embedding_dimension": "3",
            "embedding_document_prefix_fingerprint": expected_fingerprint,
            "embedding_query_prefix": "query: ",
        }
        assert owner_types == {"THOUGHT"}
        assert not vec_index_exists

    @pytest.mark.parametrize(
        ("service_override", "expected_provider"),
        [
            (False, "ollama"),
            (True, "huggingface"),
        ],
    )
    def test_re_embed_service_provider_precedence(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        service_override: bool,
        expected_provider: str,
    ) -> None:
        snap = tmp_path / f"service-source-{expected_provider}.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        data_dir = tmp_path / f"services-{expected_provider}"
        target_db = data_dir / "target.db"
        override_yaml = (
            "  configs:\n"
            "    target:\n"
            "      embeddings:\n"
            "        provider: huggingface\n"
            "        model: service-model\n"
            if service_override
            else ""
        )
        config_path = tmp_path / f"service-{expected_provider}.yaml"
        config_path.write_text(
            f"database:\n  path: {target_db}\n"
            "embeddings:\n  provider: ollama\n  model: fallback-model\n"
            f"services:\n  data_dir: {data_dir}\n  default_service: target\n"
            f"{override_yaml}",
            encoding="utf-8",
        )
        resolved_providers: list[str | None] = []

        def _resolve(config: EmbeddingConfig | None) -> _RestoreEmbeddingProvider | None:
            resolved_providers.append(config.provider if config else None)
            if config is None or config.provider is None:
                return None
            return _RestoreEmbeddingProvider(config.model or config.provider)

        monkeypatch.setattr(
            "engrava.infrastructure.service_manager.resolve_embedding_provider",
            _resolve,
        )

        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(target_db),
                "--config",
                str(config_path),
                "restore",
                "-i",
                str(snap),
                "--re-embed",
                "--service",
                "target",
            ],
        )

        assert result.exit_code == 0
        assert resolved_providers == [expected_provider]

    def test_re_embed_rejects_existing_embeddings_without_clear(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        snap = tmp_path / "nonempty-source.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        config_path = tmp_path / "nonempty.yaml"
        config_path.write_text(
            f"database:\n  path: {ms11_populated_db}\n"
            "embeddings:\n  provider: ollama\n  model: replacement-model\n",
            encoding="utf-8",
        )

        def _resolve(config: EmbeddingConfig | None) -> _RestoreEmbeddingProvider | None:
            if config is None or config.provider is None:
                return None
            return _RestoreEmbeddingProvider(config.model or config.provider)

        monkeypatch.setattr("engrava.cli.main.resolve_embedding_provider", _resolve)

        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(ms11_populated_db),
                "--config",
                str(config_path),
                "restore",
                "-i",
                str(snap),
                "--re-embed",
            ],
        )

        assert result.exit_code != 0
        assert "empty embedding target or --clear" in result.output

    def test_re_embed_no_provider_single_db(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "snap.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        new_db = tmp_path / "target.db"
        result = ms11_runner.invoke(
            cli,
            ["--db", str(new_db), "restore", "-i", str(snap), "--re-embed"],
        )
        assert result.exit_code != 0
        assert "embedding provider" in result.output.lower()

    def test_re_embed_no_provider_service_mode(
        self,
        ms11_runner: CliRunner,
        ms11_populated_db: Path,
        tmp_path: Path,
    ) -> None:
        snap = tmp_path / "snap.jsonl"
        ms11_runner.invoke(
            cli,
            ["--db", str(ms11_populated_db), "snapshot", "-o", str(snap)],
        )
        dst_dir = tmp_path / "svc-dst"

        # Create the target service first so it exists.
        async def _seed() -> None:
            async with EngravaManager(data_dir=dst_dir) as mgr:
                await mgr.get_store("target")

        asyncio.run(_seed())

        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(dst_dir / "any.db"),
                "restore",
                "-i",
                str(snap),
                "--re-embed",
                "--service",
                "target",
            ],
        )
        assert result.exit_code != 0
        assert "embedding provider" in result.output.lower()


class TestSnapshotServiceGuard:
    """Fix 2: snapshot --service must fail for non-existent service."""

    def test_snapshot_nonexistent_service_fails(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "services"
        data_dir.mkdir()
        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(data_dir / "placeholder.db"),
                "snapshot",
                "--service",
                "ghost",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
        # Must NOT have created the ghost DB.
        assert not (data_dir / "ghost.db").exists()

    def test_snapshot_existing_service_still_works(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "services"

        async def _setup() -> None:
            async with EngravaManager(data_dir=data_dir) as mgr:
                store = await mgr.get_store("real")
                t = ThoughtRecord(
                    thought_id="t-real",
                    essence="Real thought",
                    content="Content",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=1,
                    updated_cycle=1,
                )
                await store.create_thought(t)

        asyncio.run(_setup())

        out = tmp_path / "real.jsonl"
        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(data_dir / "placeholder.db"),
                "snapshot",
                "--service",
                "real",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2  # metadata + thought


class TestGetStoreConcurrency:
    """Fix 3: concurrent get_store calls must not duplicate connections."""

    async def test_concurrent_get_store_same_service(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "concurrent"
        mgr = EngravaManager(data_dir=data_dir)
        try:
            results = await asyncio.gather(
                mgr.get_store("alpha"),
                mgr.get_store("alpha"),
                mgr.get_store("alpha"),
            )
            # All must be the exact same object — single connection.
            assert results[0] is results[1]
            assert results[1] is results[2]
            # Only one DB file.
            db_files = list(data_dir.glob("*.db"))
            assert len(db_files) == 1
        finally:
            await mgr.close_all()


class TestDefaultServiceFromConfig:
    """Fix 4: CLI resolves default_service from engrava.yaml."""

    def test_snapshot_uses_default_service(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "svc"

        # Create a real service with data.
        async def _setup() -> None:
            async with EngravaManager(data_dir=data_dir) as mgr:
                store = await mgr.get_store("myapp")
                t = ThoughtRecord(
                    thought_id="t-cfg",
                    essence="Config test",
                    content="Content",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=1,
                    updated_cycle=1,
                )
                await store.create_thought(t)

        asyncio.run(_setup())

        # Write an engrava.yaml with default_service.
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {data_dir / 'myapp.db'}\n"
            f"services:\n"
            f"  data_dir: {data_dir}\n"
            f"  default_service: myapp\n",
            encoding="utf-8",
        )

        out = tmp_path / "out.jsonl"
        # No --service flag, but --config points to YAML with default_service.
        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(data_dir / "any.db"),
                "--config",
                str(cfg_file),
                "snapshot",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2  # metadata + thought

    def test_restore_uses_default_service(
        self,
        ms11_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "svc"

        # Create source service.
        async def _setup() -> None:
            async with EngravaManager(data_dir=data_dir) as mgr:
                store = await mgr.get_store("myapp")
                t = ThoughtRecord(
                    thought_id="t-restore-cfg",
                    essence="Restore config test",
                    content="Content",
                    thought_type=ThoughtType.OBSERVATION,
                    source="test",
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    priority=Priority.P2,
                    created_cycle=1,
                    updated_cycle=1,
                )
                await store.create_thought(t)

        asyncio.run(_setup())

        # Snapshot from source.
        snap = tmp_path / "snap.jsonl"
        ms11_runner.invoke(
            cli,
            [
                "--db",
                str(data_dir / "any.db"),
                "snapshot",
                "--service",
                "myapp",
                "-o",
                str(snap),
            ],
        )

        # Restore using config with default_service.
        dst_dir = tmp_path / "dst"
        cfg_file = tmp_path / "engrava.yaml"
        cfg_file.write_text(
            f"database:\n  path: {dst_dir / 'myapp.db'}\n"
            f"services:\n"
            f"  data_dir: {dst_dir}\n"
            f"  default_service: myapp\n",
            encoding="utf-8",
        )

        result = ms11_runner.invoke(
            cli,
            [
                "--db",
                str(dst_dir / "any.db"),
                "--config",
                str(cfg_file),
                "restore",
                "-i",
                str(snap),
            ],
        )
        assert result.exit_code == 0
        assert "myapp" in result.output

        # Verify data landed.
        async def _check() -> int:
            async with EngravaManager(data_dir=dst_dir) as mgr:
                store = await mgr.get_store("myapp")
                cursor = await store._db.execute("SELECT COUNT(*) FROM thought")
                row = await cursor.fetchone()
                return row[0]

        assert asyncio.run(_check()) >= 1


class TestServiceExistsMethod:
    """EngravaManager.service_exists() returns correct results."""

    async def test_service_exists_false(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "svc"
        data_dir.mkdir()
        mgr = EngravaManager(data_dir=data_dir)
        assert not mgr.service_exists("nonexistent")

    async def test_service_exists_true(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "svc"
        async with EngravaManager(data_dir=data_dir) as mgr:
            await mgr.get_store("real")
            assert mgr.service_exists("real")

    @pytest.mark.parametrize("service_name", ["../escape", "UPPER", "", "two words"])
    def test_service_exists_rejects_invalid_name(
        self,
        tmp_path: Path,
        service_name: str,
    ) -> None:
        mgr = EngravaManager(data_dir=tmp_path / "svc")
        with pytest.raises(ConfigError, match="Invalid service name"):
            mgr.service_exists(service_name)


# ------------------------------------------------------------------
# Every file the manager addresses is named by the validated name
# ------------------------------------------------------------------


class TestServiceNameNeverEscapesTheDataDirectory:
    """The name that passed the pattern check is the name that reaches the disk.

    The pattern check cannot be fooled — ``re`` reads the real text — but that
    only constrains the caller's object at the instant it is checked. Every
    filesystem operation below builds its path by interpolating a name, which
    runs ``__format__`` on whatever object it was handed, so the guarantee holds
    only while the checked value and the used value are the same value.
    """

    async def test_delete_service_leaves_files_outside_the_data_directory_alone(
        self,
        tmp_path: Path,
    ) -> None:
        """A deletion touches its own database and nothing else on the disk."""
        data_dir = tmp_path / "data"
        async with EngravaManager(data_dir=data_dir) as mgr:
            await mgr.get_store("prod")
        target = data_dir / "prod.db"
        assert target.exists()

        victim = tmp_path / "escaped.db"
        victim.write_bytes(b"not engrava's to delete")

        mgr = EngravaManager(data_dir=data_dir)
        error: Exception | None = None
        try:
            await mgr.delete_service(_LyingName("prod"))
        except Exception as exc:  # noqa: BLE001 -- the effect is asserted first
            error = exc

        # State first: a test that only asserted "something was raised" would
        # pass just as happily against code that unlinks the file and *then*
        # fails, which is the outcome this exists to rule out.
        assert victim.exists()
        assert victim.read_bytes() == b"not engrava's to delete"
        assert not target.exists()
        assert error is None

    async def test_get_store_creates_no_database_outside_the_data_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Opening a service creates its file under ``data_dir`` and nowhere else."""
        data_dir = tmp_path / "data"
        async with EngravaManager(data_dir=data_dir) as mgr:
            await mgr.get_store(_LyingName("prod"))

        assert _entry_names(tmp_path) == ["data"]
        assert (data_dir / "prod.db").exists()

    async def test_service_exists_probes_only_the_data_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Existence is answered about the named service, not an arbitrary path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (tmp_path / "escaped.db").write_bytes(b"someone else's file")

        mgr = EngravaManager(data_dir=data_dir)
        assert mgr.service_exists(_LyingName("prod")) is False

    async def test_the_same_service_opens_once_however_the_name_compares(
        self,
        tmp_path: Path,
    ) -> None:
        """The store cache is keyed on the validated name, not the caller's object."""
        async with EngravaManager(data_dir=tmp_path / "data") as mgr:
            first = await mgr.get_store("prod")
            second = await mgr.get_store(_LyingCacheKey("prod"))
            assert second is first

    async def test_legitimate_names_still_address_their_own_database(
        self,
        tmp_path: Path,
    ) -> None:
        """Ordinary names round-trip through create, exist, and delete unchanged."""
        data_dir = tmp_path / "data"
        names = ("main", "home-assistant", "coding_helper", "a1b2")
        async with EngravaManager(data_dir=data_dir) as mgr:
            for name in names:
                await mgr.get_store(name)
                assert mgr.service_exists(name)
            assert await mgr.list_services() == sorted(names)

        mgr = EngravaManager(data_dir=data_dir)
        await mgr.delete_service("main")
        assert await mgr.list_services() == sorted(names[1:])


class TestServicesConfigOwnsItsNames:
    """``ServicesConfig`` stores the names it validated, as plain strings."""

    def test_default_service_is_an_exact_str(self) -> None:
        cfg = ServicesConfig(data_dir=Path("./data"), default_service=_LyingName("prod"))
        assert type(cfg.default_service) is str
        assert cfg.default_service == "prod"

    def test_config_keys_are_the_validated_names(self) -> None:
        cfg = ServicesConfig(
            data_dir=Path("./data"),
            default_service="main",
            configs={_LyingCacheKey("prod"): ServiceConfig()},
        )
        assert [type(key) for key in cfg.configs] == [str]
        assert cfg.configs["prod"] == ServiceConfig()


def _look_alike_of(config: object) -> object:
    """Return *config* with its ``__class__`` reassigned to a non-subclass."""

    class _LookAlike:
        pass

    object.__setattr__(config, "__class__", _LookAlike)
    return config


class TestConfigObjectsMustBeExactlyTheirClass:
    """The store and the manager retain configuration and read it for themselves.

    Each is a public entry point, so each requires the exact class at its own
    boundary rather than assuming the other did. Subclassing a configuration is
    refused at the class now, so what still has to be caught here is a
    ``__class__`` reassigned to a look-alike that was never a subclass, and
    anything that was never a configuration at all.
    """

    async def test_the_store_refuses_a_look_alike_hygiene_policy(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            with pytest.raises(ConfigError, match="must be exactly a HygienePolicyConfig"):
                SqliteEngravaCore(conn, hygiene_policy=_look_alike_of(HygienePolicyConfig()))
        finally:
            await conn.close()

    def test_a_hygiene_policy_cannot_be_subclassed_in_the_first_place(self) -> None:
        """The two-faced policy that reopened this needed a subclass to exist."""
        with pytest.raises(TypeError, match="may not be subclassed"):
            type("_TwoFacedPolicy", (HygienePolicyConfig,), {})

    @pytest.mark.parametrize(
        ("kwarg", "cls"),
        [
            ("search_config", SearchConfig),
            ("metrics_config", MetricsConfig),
            ("hygiene_policy", HygienePolicyConfig),
            ("derive_gates", DeriveGates),
        ],
    )
    async def test_every_config_the_store_retains_requires_its_exact_class(
        self,
        kwarg: str,
        cls: type,
    ) -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            with pytest.raises(ConfigError, match=f"must be exactly a {cls.__name__}"):
                SqliteEngravaCore(conn, **{kwarg: _look_alike_of(cls())})
        finally:
            await conn.close()

    @pytest.mark.parametrize(
        ("kwarg", "cls"),
        [
            ("default_embeddings", EmbeddingConfig),
            ("default_search", SearchConfig),
        ],
    )
    def test_every_config_the_manager_retains_requires_its_exact_class(
        self,
        tmp_path: Path,
        kwarg: str,
        cls: type,
    ) -> None:
        with pytest.raises(ConfigError, match=f"must be exactly a {cls.__name__}"):
            EngravaManager(data_dir=tmp_path / "data", **{kwarg: _look_alike_of(cls())})

    def test_the_manager_refuses_a_look_alike_services_config(self, tmp_path: Path) -> None:
        services = _look_alike_of(ServicesConfig(data_dir=tmp_path / "data"))
        with pytest.raises(ConfigError, match="must be exactly a ServicesConfig"):
            EngravaManager(data_dir=tmp_path / "data", services_config=services)

    async def test_the_exact_classes_are_still_accepted(self, tmp_path: Path) -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            store = SqliteEngravaCore(
                conn,
                search_config=SearchConfig(),
                metrics_config=MetricsConfig(),
                hygiene_policy=HygienePolicyConfig(),
                derive_gates=DeriveGates(),
            )
            assert store is not None
        finally:
            await conn.close()

        async with EngravaManager(
            data_dir=tmp_path / "data",
            default_search=SearchConfig(),
            services_config=ServicesConfig(data_dir=tmp_path / "data"),
        ) as manager:
            assert await manager.get_store("main") is not None


class TestStoreConstructorParametersAreOwned:
    """The store takes raw numbers straight from its caller, not only via config.

    The configuration sweep never sees these, which is why the whole class of
    defect reappeared here one layer out. The cadence in particular decides
    whether an automatic cleanup runs at all, and under the ``delete`` strategy
    that cleanup destroys rows.
    """

    async def test_a_cadence_that_denies_its_own_value_deletes_nothing(self) -> None:
        """Configured off, it stays off — whatever the value answers to ``< 1``."""

        class _CadenceThatDeniesItsOwnValue(int):
            __slots__ = ()

            def __lt__(self, other: object) -> bool:
                del other
                return False

        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(
                conn,
                ttl_strategy="delete",
                ttl_check_every_n=_CadenceThatDeniesItsOwnValue(0),
            )
            await store.ensure_schema()
            expired = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)).isoformat()
            await store.create_thought(
                ThoughtRecord(
                    thought_id="victim",
                    thought_type=ThoughtType.OBSERVATION,
                    essence="e",
                    content="c",
                    priority=Priority.P3,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    source="test",
                    expires_at=expired,
                )
            )
            await store.create_thought(
                ThoughtRecord(
                    thought_id="other",
                    thought_type=ThoughtType.OBSERVATION,
                    essence="e",
                    content="c",
                    priority=Priority.P3,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    source="test",
                )
            )

            # State first: the row is the question. An automatic cleanup the
            # caller switched off must not have run at all.
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM thought WHERE thought_id = ?", ("victim",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1
        finally:
            await conn.close()

    async def test_the_stored_cadence_is_an_exact_int(self) -> None:
        class _Cadence(int):
            __slots__ = ()

        conn = await aiosqlite.connect(":memory:")
        try:
            store = SqliteEngravaCore(conn, ttl_check_every_n=_Cadence(5))
            assert type(store._ttl_check_every_n) is int
            assert store._ttl_check_every_n == 5
        finally:
            await conn.close()

    @pytest.mark.parametrize("kwarg", ["ttl_check_every_n", "ttl_default_seconds"])
    async def test_a_non_integer_is_a_configuration_error(self, kwarg: str) -> None:
        conn = await aiosqlite.connect(":memory:")
        try:
            with pytest.raises(ConfigError, match="must be an integer"):
                SqliteEngravaCore(conn, **{kwarg: "5"})
        finally:
            await conn.close()

    async def test_a_legitimate_cadence_still_runs_cleanup(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        try:
            store = SqliteEngravaCore(conn, ttl_strategy="delete", ttl_check_every_n=1)
            await store.ensure_schema()
            expired = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)).isoformat()
            await store.create_thought(
                ThoughtRecord(
                    thought_id="victim",
                    thought_type=ThoughtType.OBSERVATION,
                    essence="e",
                    content="c",
                    priority=Priority.P3,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    source="test",
                    expires_at=expired,
                )
            )
            await store.create_thought(
                ThoughtRecord(
                    thought_id="other",
                    thought_type=ThoughtType.OBSERVATION,
                    essence="e",
                    content="c",
                    priority=Priority.P3,
                    lifecycle_status=LifecycleStatus.ACTIVE,
                    source="test",
                )
            )
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM thought WHERE thought_id = ?", ("victim",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0
        finally:
            await conn.close()

    def test_the_manager_stores_an_exact_embedding_dimension(self, tmp_path: Path) -> None:
        class _Dimension(int):
            __slots__ = ()

        manager = EngravaManager(data_dir=tmp_path / "data", embedding_dimension=_Dimension(384))
        assert type(manager._embedding_dimension) is int
