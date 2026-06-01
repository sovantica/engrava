-- engrava: Core thought-graph schema (free-tier boundary — no internal-cognitive columns).
-- Version: core-12 (referential integrity: FK + ON DELETE CASCADE on edge/embedding/action → thought)

PRAGMA user_version = 12;

CREATE TABLE IF NOT EXISTS thought (
    thought_id        TEXT    PRIMARY KEY,
    thought_type      TEXT    NOT NULL,
    essence           TEXT    NOT NULL,
    content           TEXT    NOT NULL,
    content_hash      TEXT,
    priority          TEXT    NOT NULL,
    lifecycle_status  TEXT    NOT NULL DEFAULT 'CREATED',
    created_cycle     INTEGER NOT NULL DEFAULT 0,
    updated_cycle     INTEGER NOT NULL DEFAULT 0,
    source            TEXT    NOT NULL DEFAULT 'human',
    confidence        REAL,
    embedding_ref     TEXT,
    source_type       TEXT    NOT NULL DEFAULT 'EXPERIENCE',
    confirmation_count INTEGER NOT NULL DEFAULT 0,
    consolidated_from TEXT,
    visibility        TEXT    NOT NULL DEFAULT 'selective',
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_accessed_at  TEXT,
    created_at        TEXT,
    updated_at        TEXT,
    expires_at        TEXT,
    metadata_json     TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edge (
    edge_id           TEXT PRIMARY KEY,
    from_thought_id   TEXT NOT NULL,
    to_thought_id     TEXT NOT NULL,
    edge_type         TEXT NOT NULL,
    weight            REAL NOT NULL DEFAULT 0.5,
    created_cycle     INTEGER NOT NULL DEFAULT 0,
    source            TEXT NOT NULL DEFAULT 'EXPERIENCE',
    decay_multiplier  REAL NOT NULL DEFAULT 1.0,
    UNIQUE(from_thought_id, to_thought_id, edge_type),
    FOREIGN KEY (from_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE,
    FOREIGN KEY (to_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE
);

-- ``embedding.owner_id`` references ``thought(thought_id)`` because every
-- embedding ingested by this codebase is keyed to a thought
-- (``owner_type='THOUGHT'`` on every insert). The polymorphic-looking
-- ``owner_type`` column is reserved for forward-compatibility but is not
-- currently exercised by any non-thought owner; the FK + CASCADE assumption
-- here is safe today and any future non-thought owner would have to revisit
-- this constraint.
CREATE TABLE IF NOT EXISTS embedding (
    embedding_id TEXT PRIMARY KEY,
    owner_type   TEXT    NOT NULL,
    owner_id     TEXT    NOT NULL,
    model_name   TEXT    NOT NULL,
    dimension    INTEGER NOT NULL,
    vector_blob  BLOB    NOT NULL,
    created_at   TEXT    NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES thought(thought_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS action (
    action_id           TEXT PRIMARY KEY,
    source_thought_id   TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    intent              TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'PLANNED',
    verification_status TEXT NOT NULL DEFAULT 'PENDING',
    raw_metrics_json    TEXT,
    FOREIGN KEY (source_thought_id) REFERENCES thought(thought_id) ON DELETE CASCADE
);

-- -------------------------------------------------------------------
-- Generic key/value metadata
-- -------------------------------------------------------------------
-- Used for embedding model lock (embedding_model_name, embedding_dimension)
-- and future migration metadata (schema_version, created_at, etc.).

CREATE TABLE IF NOT EXISTS _metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- -------------------------------------------------------------------
-- FTS5 full-text search index for topic-based retrieval
-- -------------------------------------------------------------------
-- content= syncs with the thought table via triggers.
-- content_rowid='rowid' uses the implicit rowid (thought is NOT
-- WITHOUT ROWID, so the implicit rowid column exists).
-- Only essence and content are indexed — sufficient for keyword recall.

CREATE VIRTUAL TABLE IF NOT EXISTS thought_fts USING fts5(
    essence,
    content,
    tokenize = "unicode61 tokenchars '-_'",
    content='thought',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS thought_fts_insert
AFTER INSERT ON thought BEGIN
    INSERT INTO thought_fts(rowid, essence, content)
    VALUES (new.rowid, new.essence, new.content);
END;

CREATE TRIGGER IF NOT EXISTS thought_fts_delete
AFTER DELETE ON thought BEGIN
    INSERT INTO thought_fts(thought_fts, rowid, essence, content)
    VALUES ('delete', old.rowid, old.essence, old.content);
END;

CREATE TRIGGER IF NOT EXISTS thought_fts_update
AFTER UPDATE OF essence, content ON thought BEGIN
    INSERT INTO thought_fts(thought_fts, rowid, essence, content)
    VALUES ('delete', old.rowid, old.essence, old.content);
    INSERT INTO thought_fts(rowid, essence, content)
    VALUES (new.rowid, new.essence, new.content);
END;

-- -------------------------------------------------------------------
-- CognitiveJournal: hash-linked mutation log
-- -------------------------------------------------------------------
-- Append-only audit trail for thought-graph mutations.  Each entry is
-- SHA-256 linked to the previous entry (parent_hash) for tamper evidence.

CREATE TABLE IF NOT EXISTS journal_entry (
    entry_id         TEXT    PRIMARY KEY,
    sequence_number  INTEGER NOT NULL UNIQUE,
    mutation_type    TEXT    NOT NULL,
    target_id        TEXT,
    delta            TEXT    NOT NULL,
    parent_hash      TEXT,
    entry_hash       TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_target ON journal_entry(target_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entry(mutation_type);
CREATE INDEX IF NOT EXISTS idx_journal_seq ON journal_entry(sequence_number);

-- -------------------------------------------------------------------
-- Partial index for TTL / auto-expiry
-- -------------------------------------------------------------------
-- Only thoughts with a non-NULL expires_at are indexed.  Thoughts that
-- never expire (the vast majority) incur zero index overhead.

CREATE INDEX IF NOT EXISTS idx_thought_expires ON thought(expires_at)
    WHERE expires_at IS NOT NULL;

-- -------------------------------------------------------------------
-- Content-hash index for ingest deduplication (core-10)
-- -------------------------------------------------------------------
-- Enables O(log N) lookup of existing thoughts by SHA-256 content hash
-- when callers opt in to deduplication via
-- ``SqliteEngravaCore.create_thought(..., deduplicate=True)``.
-- Nullable column: pre-core-10 thoughts may have ``content_hash IS NULL``
-- until the explicit backfill utility script populates them.

CREATE INDEX IF NOT EXISTS idx_thought_content_hash ON thought(content_hash);

-- -------------------------------------------------------------------
-- Composite edge index for candidate expansion queries
-- -------------------------------------------------------------------
-- Supports _expand_via_consolidated_from and the giant-cluster guard:
--   SELECT ... FROM edge
--   WHERE edge_type = 'CONSOLIDATED_FROM' AND from_thought_id IN (...)
-- Also benefits _load_graph_signal and any future edge_type lookups.

CREATE INDEX IF NOT EXISTS idx_edge_type_from ON edge(edge_type, from_thought_id);

-- -------------------------------------------------------------------
-- Per-extension schema version tracking
-- -------------------------------------------------------------------
-- Tracks which SQL migration files have been applied for each extension.
-- extension_name matches ExtensionManifest.name.
-- version is the count of applied migration files (1-indexed).

CREATE TABLE IF NOT EXISTS extension_schema_versions (
    extension_name    TEXT PRIMARY KEY,
    version           INTEGER NOT NULL DEFAULT 0,
    applied_at        REAL NOT NULL,
    migration_file    TEXT,
    extension_version TEXT
);
