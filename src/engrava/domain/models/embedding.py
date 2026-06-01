"""EmbeddingRecord — vector storage for semantic retrieval."""

from __future__ import annotations

import datetime
import struct
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EmbeddingRecord(BaseModel):
    """Vector storage record for semantic retrieval.

    Args:
        embedding_id: Stable UUID identity.
        owner_type: Entity type that owns this embedding (e.g., 'THOUGHT').
        owner_id: UUID of the owning entity.
        model_name: Embedding model identifier (e.g., 'all-MiniLM-L12-v2').
        dimension: Vector dimensionality (must match vector_blob length).
        vector_blob: Serialized embedding vector as bytes.
        created_at: ISO-8601 timestamp of creation.

    Examples:
        >>> import struct
        >>> vector = struct.pack("3f", 0.1, 0.2, 0.3)
        >>> record = EmbeddingRecord(
        ...     embedding_id="emb-001",
        ...     owner_type="THOUGHT",
        ...     owner_id="thought-001",
        ...     model_name="all-MiniLM-L12-v2",
        ...     dimension=3,
        ...     vector_blob=vector,
        ...     created_at="2026-03-11T00:00:00Z",
        ... )

    """

    model_config = ConfigDict(frozen=True)

    embedding_id: str
    owner_type: str = Field(min_length=1)
    owner_id: str
    model_name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    vector_blob: bytes
    created_at: str = Field(min_length=1)

    @field_validator("embedding_id", "owner_id")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        """Validate that ID fields are not empty or whitespace."""
        if not v.strip():
            msg = "ID field must not be empty or whitespace"
            raise ValueError(msg)
        return v

    @field_validator("created_at")
    @classmethod
    def _validate_created_at_iso8601(cls, v: str) -> str:
        """Validate that created_at is a valid ISO-8601 timestamp."""
        try:
            datetime.datetime.fromisoformat(v)
        except ValueError:
            msg = f"created_at must be an ISO-8601 timestamp, got {v!r}"
            raise ValueError(msg) from None
        return v

    @model_validator(mode="after")
    def _validate_dimension_matches_vector(self) -> Self:
        """Ensure dimension matches the actual vector blob length."""
        float_size = struct.calcsize("f")
        expected_bytes = self.dimension * float_size
        if len(self.vector_blob) != expected_bytes:
            msg = (
                f"vector_blob length ({len(self.vector_blob)} bytes) does not match "
                f"dimension ({self.dimension}) x {float_size} bytes/float = "
                f"{expected_bytes} bytes"
            )
            raise ValueError(msg)
        return self
