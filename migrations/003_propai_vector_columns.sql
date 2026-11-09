-- Empty pgvector columns for Prop-AI name and text tables.
-- 384 dimensions matches all-MiniLM-L6-v2.
-- Sales, rents, images, and sync state stay without an embedding column.
-- A superuser must run this once first: CREATE EXTENSION vector;

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.propai_places
  ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE public.propai_raw
  ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE public.propai_brokers
  ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE public.propai_project_status
  ADD COLUMN IF NOT EXISTS embedding vector(384);
