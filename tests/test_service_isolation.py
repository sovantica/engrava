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
    LifecycleStatus,
    Priority,
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
from engrava.infrastructure.service_manager import EngravaManager

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
            _validate_service_name(name)  # should not raise

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
            assert row_a[0] == row_b[0] == 12

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
        assert header["schema_version"] == 12
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
