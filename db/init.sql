-- Postgres bootstrap (runs once via docker-entrypoint-initdb.d).
--
-- The relational app tables (developers, evidence_artifacts, skill_records,
-- skill_evidence_links, plus the API tables) are owned by SQLAlchemy/Alembic so
-- there's ONE source of truth and dev (sqlite) matches prod. This file only sets
-- up the Postgres-specific bits the ORM can't express portably: the extensions
-- and the pgvector embedding store used for semantic evidence search.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Semantic search over evidence summaries. artifact_id references
-- evidence_artifacts(id) by convention (no FK here because that table is created
-- later by the app, after this init script has already run on the empty DB).
CREATE TABLE IF NOT EXISTS embedding_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_id VARCHAR(32) UNIQUE NOT NULL,
    embedding VECTOR(384),
    model_version VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_vector ON embedding_records
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
