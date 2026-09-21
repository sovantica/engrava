"""engrava — standalone thought-graph persistence engine."""

from engrava.config import (
    DreamingConfig,
    DreamingGates,
    EmbeddingConfig,
    EngravaConfig,
    HygienePolicyConfig,
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
from engrava.config_validation import ConfigError
from engrava.cycle_providers import (
    CallableCycleProvider,
    MaxCycleProvider,
    StaticCycleProvider,
)
from engrava.domain.dreaming import (
    ActionOutcomeSignal,
    ConfidenceSignal,
    ConfirmationSignal,
    ConsolidationResult,
    DreamingContext,
    DreamingSignalProtocol,
    FrequencySignal,
    RecencySignal,
    StalenessSignal,
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
    ConnectionQuarantinedError,
    CoreMigrationError,
    CycleProviderError,
    DedupLockReentryError,
    DerivedRecordError,
    DuplicateEdgeError,
    EmbeddingGenerationError,
    EmbeddingModelMismatchError,
    EmbeddingProviderContractError,
    EmbeddingQueryPrefixMismatchError,
    EngravaError,
    ExtensionMigrationError,
    InvalidFilterError,
    InvalidFilterPathError,
    InvalidRecencyArgumentError,
    InvalidTransitionError,
    JournalIntegrityError,
    ReadOnlyViolationError,
    RecencyModeConflictError,
    ReferentialIntegrityError,
    SchemaVersionError,
    SourceThoughtNotFoundError,
    StaleDataError,
    ThoughtNotFoundError,
    VectorDimensionMismatchError,
    WriteContentionError,
    WriteLockTimeoutError,
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
from engrava.domain.models.provenance import ProvenanceContext
from engrava.domain.models.search import HybridSearchResult
from engrava.domain.models.thought import ThoughtRecord
from engrava.domain.models.thought import ThoughtRecord as CoreThoughtRecord
from engrava.domain.models.ttl import CleanupResult, CleanupStrategy
from engrava.domain.protocols.cycle_provider import CycleProvider
from engrava.domain.protocols.derived_records import (
    DeriveContext,
    DerivedRecord,
    DerivedRecordProducerProtocol,
    DeriveGates,
    DeriveResult,
)
from engrava.domain.protocols.embedding_provider import (
    EmbeddingProviderProtocol,
    RoleAwareEmbeddingProvider,
)
from engrava.domain.protocols.engrava_core import EngravaCoreProtocol
from engrava.domain.protocols.engrava_read import EngravaReadProtocol
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
from engrava.extensions.dreaming import DreamingExtension
from engrava.extensions.structural_split import SplitMode, StructuralSplitProducer
from engrava.infrastructure.read_only_store import ReadOnlyEngrava
from engrava.infrastructure.service_manager import EngravaManager
from engrava.infrastructure.sqlite.engrava_core import SqliteEngravaCore
from engrava.infrastructure.sqlite.hygiene import EvictionReason, HygieneResult
from engrava.infrastructure.sqlite.journal_writer import JournalWriter
from engrava.infrastructure.sqlite.vector_sqlite_vec import SqliteVecSearchBackend
from engrava.metadata import percept, thought, utterance
from engrava.mindql.executor import MindQLExecutor, MindQLResult
from engrava.mindql.parser import MindQLCommand, MindQLParseError, MindQLQuery, parse

__all__ = [
    "ActionNotFoundError",
    "ActionOutcomeSignal",
    "ActionRecord",
    "ActionStatus",
    "ActionType",
    "CallableCycleProvider",
    "CallbackProvider",
    "CleanupResult",
    "CleanupStrategy",
    "ConfidenceSignal",
    "ConfigError",
    "ConfirmationSignal",
    "ConnectionQuarantinedError",
    "ConsolidationResult",
    "CoreMigrationError",
    "CoreThoughtRecord",
    "CycleProvider",
    "CycleProviderError",
    "DedupLockReentryError",
    "DefaultEngravaHooks",
    "DefaultMindStoreHooks",
    "DeriveContext",
    "DeriveGates",
    "DeriveResult",
    "DerivedRecord",
    "DerivedRecordError",
    "DerivedRecordProducerProtocol",
    "DreamingConfig",
    "DreamingContext",
    "DreamingExtension",
    "DreamingGates",
    "DreamingSignalProtocol",
    "DuplicateEdgeError",
    "EdgeCounts",
    "EdgeRecord",
    "EdgeType",
    "EmbeddingConfig",
    "EmbeddingGenerationError",
    "EmbeddingModelMismatchError",
    "EmbeddingProviderContractError",
    "EmbeddingProviderProtocol",
    "EmbeddingQueryPrefixMismatchError",
    "EmbeddingRecord",
    "EngravaConfig",
    "EngravaCoreProtocol",
    "EngravaError",
    "EngravaHooksProtocol",
    "EngravaManager",
    "EngravaMetrics",
    "EngravaReadProtocol",
    "EvictionReason",
    "ExtensionManifest",
    "ExtensionMigrationError",
    "FieldOp",
    "FieldPredicate",
    "FrequencySignal",
    "HuggingFaceProvider",
    "HybridSearchResult",
    "HygienePolicyConfig",
    "HygieneResult",
    "InvalidFilterError",
    "InvalidFilterPathError",
    "InvalidRecencyArgumentError",
    "InvalidTransitionError",
    "JournalConfig",
    "JournalEntry",
    "JournalIntegrityError",
    "JournalIntegrityResult",
    "JournalWriter",
    "KnowledgeSource",
    "LatencyHistogram",
    "LifecycleStatus",
    "MaxCycleProvider",
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
    "ProvenanceContext",
    "ReadOnlyEngrava",
    "ReadOnlyMindStore",
    "ReadOnlyViolationError",
    "RecencyModeConflictError",
    "RecencySignal",
    "ReferentialIntegrityError",
    "RoleAwareEmbeddingProvider",
    "SchemaVersionError",
    "ScoringContext",
    "SearchConfig",
    "SentenceTransformerProvider",
    "ServiceConfig",
    "ServicesConfig",
    "SourceThoughtNotFoundError",
    "SplitMode",
    "SqliteEngravaCore",
    "SqliteMindStoreCore",
    "SqliteVecSearchBackend",
    "StaleDataError",
    "StalenessSignal",
    "StaticCycleProvider",
    "StorageFootprint",
    "StructuralSplitProducer",
    "TTLConfig",
    "ThoughtCounts",
    "ThoughtNotFoundError",
    "ThoughtRecord",
    "ThoughtType",
    "ThoughtVisibility",
    "VectorDimensionMismatchError",
    "VerificationStatus",
    "VisibilityQueryFilter",
    "WriteContentionError",
    "WriteLockTimeoutError",
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
# Backward-compatibility aliases for the pre-rename (MindStore-era) names.
#
# Policy: kept, not scheduled for removal. A caller may keep using these
# names — removing them would be a breaking change, and under this
# project's versioning that computes a major release, which is not a
# price this surface is worth paying on its own. Keeping them costs one
# attribute lookup (the __getattr__ below) and no ongoing maintenance.
# This is not a promise they live forever; it is the current position,
# to be revisited deliberately rather than left to expire silently. New
# code should use the current names, which is what the DeprecationWarning
# on access points at.
# ------------------------------------------------------------------
import warnings as _warnings

# The source of truth for every pre-rename alias this module still serves.
# Tests reach into this constant (rather than hand-copying it) so that an
# alias added here without test coverage fails loudly instead of shipping
# silently.
_DEPRECATED_ALIASES: dict[str, object] = {
    "SqliteMindStoreCore": SqliteEngravaCore,
    "MindStoreManager": EngravaManager,
    "MindStoreConfig": EngravaConfig,
    "MindStoreError": EngravaError,
    "MindStoreCoreProtocol": EngravaCoreProtocol,
    "MindStoreHooksProtocol": EngravaHooksProtocol,
    "DefaultMindStoreHooks": DefaultEngravaHooks,
    "ReadOnlyMindStore": ReadOnlyEngrava,
}


def __getattr__(name: str) -> object:
    """Lazy deprecation aliases for renamed symbols."""
    if name in _DEPRECATED_ALIASES:
        target = _DEPRECATED_ALIASES[name]
        target_name = getattr(target, "__name__", str(target))
        _warnings.warn(
            f"{name} is deprecated, use {target_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return target
    msg = f"module 'engrava' has no attribute {name!r}"
    raise AttributeError(msg)
