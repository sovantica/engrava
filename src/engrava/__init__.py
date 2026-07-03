"""engrava — standalone thought-graph persistence engine."""

from engrava.config import (
    ConfigError,
    DreamingConfig,
    DreamingGates,
    EmbeddingConfig,
    EngravaConfig,
    JournalConfig,
    MetricsConfig,
    SearchConfig,
    ServiceConfig,
    ServicesConfig,
    TTLConfig,
    load_config,
    resolve_embedding_provider,
    resolve_hooks,
    resolve_manifests,
)
from engrava.domain.enums import (
    ActionStatus,
    ActionType,
    EdgeType,
    KnowledgeSource,
    LifecycleStatus,
    Priority,
    ThoughtType,
    ThoughtVisibility,
    VerificationStatus,
)
from engrava.domain.exceptions import (
    ActionNotFoundError,
    EmbeddingGenerationError,
    EmbeddingModelMismatchError,
    EmbeddingQueryPrefixMismatchError,
    EngravaError,
    ExtensionMigrationError,
    InvalidFilterError,
    InvalidFilterPathError,
    InvalidTransitionError,
    JournalIntegrityError,
    ReadOnlyViolationError,
    StaleDataError,
    ThoughtNotFoundError,
)
from engrava.domain.manifest import ExtensionManifest
from engrava.domain.models.action import ActionRecord
from engrava.domain.models.edge import EdgeRecord
from engrava.domain.models.embedding import EmbeddingRecord
from engrava.domain.models.filters import (
    FieldOp,
    FieldPredicate,
    MetadataFilter,
    VisibilityQueryFilter,
)
from engrava.domain.models.journal import JournalEntry, JournalIntegrityResult
from engrava.domain.models.metrics import (
    EdgeCounts,
    EngravaMetrics,
    LatencyHistogram,
    StorageFootprint,
    ThoughtCounts,
)
from engrava.domain.models.mutation_type import MutationType
from engrava.domain.models.search import HybridSearchResult
from engrava.domain.models.thought import ThoughtRecord
from engrava.domain.models.thought import ThoughtRecord as CoreThoughtRecord
from engrava.domain.models.ttl import CleanupResult, CleanupStrategy
from engrava.domain.protocols.embedding_provider import (
    EmbeddingProviderProtocol,
    RoleAwareEmbeddingProvider,
)
from engrava.domain.protocols.engrava_core import EngravaCoreProtocol
from engrava.domain.protocols.hooks import (
    DefaultEngravaHooks,
    EngravaHooksProtocol,
    MindQLExtension,
    ScoringContext,
)
from engrava.embeddings.callback import CallbackProvider
from engrava.embeddings.huggingface import HuggingFaceProvider
from engrava.embeddings.ollama import OllamaProvider
from engrava.embeddings.openai_compatible import OpenAICompatibleProvider
from engrava.embeddings.sentence_transformer import SentenceTransformerProvider
from engrava.extensions.discovery import discover_manifests
from engrava.extensions.dreaming import ConsolidationResult, DreamingExtension
from engrava.extensions.dreaming_signals import (
    ActionOutcomeSignal,
    ConfidenceSignal,
    ConfirmationSignal,
    DreamingContext,
    DreamingSignalProtocol,
    FrequencySignal,
    RecencySignal,
    StalenessSignal,
)
from engrava.extensions.vector_sqlite_vec import SqliteVecSearchBackend
from engrava.infrastructure.read_only_store import ReadOnlyEngrava
from engrava.infrastructure.service_manager import EngravaManager
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
from engrava.infrastructure.sqlite.journal_writer import JournalWriter
from engrava.metadata import percept, thought, utterance
from engrava.mindql.executor import MindQLExecutor, MindQLResult
from engrava.mindql.parser import MindQLCommand, MindQLParseError, MindQLQuery, parse

__all__ = [
    "ActionNotFoundError",
    "ActionOutcomeSignal",
    "ActionRecord",
    "ActionStatus",
    "ActionType",
    "CallbackProvider",
    "CleanupResult",
    "CleanupStrategy",
    "ConfidenceSignal",
    "ConfigError",
    "ConfirmationSignal",
    "ConsolidationResult",
    "CoreThoughtRecord",
    "DefaultEngravaHooks",
    "DefaultMindStoreHooks",
    "DreamingConfig",
    "DreamingContext",
    "DreamingExtension",
    "DreamingGates",
    "DreamingSignalProtocol",
    "EdgeCounts",
    "EdgeRecord",
    "EdgeType",
    "EmbeddingConfig",
    "EmbeddingGenerationError",
    "EmbeddingModelMismatchError",
    "EmbeddingProviderProtocol",
    "EmbeddingQueryPrefixMismatchError",
    "EmbeddingRecord",
    "EngravaConfig",
    "EngravaCoreProtocol",
    "EngravaError",
    "EngravaHooksProtocol",
    "EngravaManager",
    "EngravaMetrics",
    "ExtensionManifest",
    "ExtensionMigrationError",
    "FieldOp",
    "FieldPredicate",
    "FrequencySignal",
    "HuggingFaceProvider",
    "HybridSearchResult",
    "InvalidFilterError",
    "InvalidFilterPathError",
    "InvalidTransitionError",
    "JournalConfig",
    "JournalEntry",
    "JournalIntegrityError",
    "JournalIntegrityResult",
    "JournalWriter",
    "KnowledgeSource",
    "LatencyHistogram",
    "LifecycleStatus",
    "MetadataFilter",
    "MetricsConfig",
    "MindQLCommand",
    "MindQLExecutor",
    "MindQLExtension",
    "MindQLParseError",
    "MindQLQuery",
    "MindQLResult",
    "MindStoreConfig",
    "MindStoreCoreProtocol",
    "MindStoreError",
    "MindStoreHooksProtocol",
    "MindStoreManager",
    "MutationType",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "Priority",
    "ReadOnlyEngrava",
    "ReadOnlyMindStore",
    "ReadOnlyViolationError",
    "RecencySignal",
    "RoleAwareEmbeddingProvider",
    "ScoringContext",
    "SearchConfig",
    "SentenceTransformerProvider",
    "ServiceConfig",
    "ServicesConfig",
    "SqliteEngravaCore",
    "SqliteMindStoreCore",
    "SqliteVecSearchBackend",
    "StaleDataError",
    "StalenessSignal",
    "StorageFootprint",
    "TTLConfig",
    "ThoughtCounts",
    "ThoughtNotFoundError",
    "ThoughtRecord",
    "ThoughtType",
    "ThoughtVisibility",
    "VerificationStatus",
    "VisibilityQueryFilter",
    "discover_manifests",
    "load_config",
    "parse",
    "percept",
    "resolve_embedding_provider",
    "resolve_hooks",
    "resolve_manifests",
    "thought",
    "utterance",
]


# ------------------------------------------------------------------
# Backward-compatibility aliases — deprecated, remove in v0.4
# ------------------------------------------------------------------
import warnings as _warnings


def __getattr__(name: str) -> object:
    """Lazy deprecation aliases for renamed symbols."""
    _aliases: dict[str, object] = {
        "SqliteMindStoreCore": SqliteEngravaCore,
        "MindStoreManager": EngravaManager,
        "MindStoreConfig": EngravaConfig,
        "MindStoreError": EngravaError,
        "MindStoreCoreProtocol": EngravaCoreProtocol,
        "MindStoreHooksProtocol": EngravaHooksProtocol,
        "DefaultMindStoreHooks": DefaultEngravaHooks,
        "ReadOnlyMindStore": ReadOnlyEngrava,
    }
    if name in _aliases:
        target = _aliases[name]
        target_name = getattr(target, "__name__", str(target))
        _warnings.warn(
            f"{name} is deprecated, use {target_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return target
    msg = f"module 'engrava' has no attribute {name!r}"
    raise AttributeError(msg)
