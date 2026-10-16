-- =====================================================================
-- 001_pgvector.sql
-- Convert the public-schema embedding columns from real[] to pgvector
-- and switch the match_* functions to pgvector's cosine distance.
--
-- PREREQUISITE (a database superuser must do this once, on the server):
--     sudo apt install postgresql-18-pgvector
--     CREATE EXTENSION vector;      -- run inside database acotdb
--
-- The whole script runs in ONE transaction: if anything fails, nothing
-- is changed. Take a backup first:
--     pg_dump -h HOST -U USER -d acotdb -n public -Fc -f acotdb_public_before_pgvector.dump
--
-- The match_* functions keep their exact signature (query_embedding real[])
-- and return the same JSON fields, so the Python code does not change.
-- cosine_sim() is left in place (used by the rollback script).
-- =====================================================================

BEGIN;

-- ---------------------------------------------------------------------
-- 0. Safety check
-- ---------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    RAISE EXCEPTION
      'pgvector is not installed in this database. A superuser must run: CREATE EXTENSION vector;';
  END IF;
END $$;

-- ---------------------------------------------------------------------
-- 1. Convert embedding columns: real[] -> vector(384)
--    (fails safely if any row is not exactly 384 dimensions)
-- ---------------------------------------------------------------------
ALTER TABLE public.communities     ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384);
ALTER TABLE public.sub_communities ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384);
ALTER TABLE public.projects        ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384);
ALTER TABLE public.document_chunks ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384);

-- ---------------------------------------------------------------------
-- 2. HNSW indexes for cosine similarity
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS communities_embedding_hnsw
  ON public.communities     USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS sub_communities_embedding_hnsw
  ON public.sub_communities USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS projects_embedding_hnsw
  ON public.projects        USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
  ON public.document_chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- 3. match_* functions using pgvector (<=> is cosine distance)
--    similarity = 1 - cosine distance, same meaning as before.
--    ORDER BY ... LIMIT lets PostgreSQL use the HNSW index; the
--    threshold is applied to those top rows.
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.match_communities(
  query_embedding real[],
  match_threshold double precision DEFAULT 0.7,
  match_count integer DEFAULT 12)
RETURNS SETOF jsonb
LANGUAGE sql STABLE
AS $function$
  select jsonb_build_object('id', id, 'slug', slug, 'name', name, 'city', city, 'photos', photos,
                            'total_count', total_count, 'knowledge_text', knowledge_text, 'similarity', sim)
  from (
    select c.id, c.slug, c.name, c.city, c.photos, c.total_count, c.knowledge_text,
           1 - (c.embedding <=> query_embedding::vector(384)) as sim
    from public.communities c
    where c.embedding is not null
    order by c.embedding <=> query_embedding::vector(384)
    limit match_count
  ) s
  where sim > match_threshold
  order by sim desc
$function$;

CREATE OR REPLACE FUNCTION public.match_projects(
  query_embedding real[],
  match_threshold double precision DEFAULT 0.7,
  match_count integer DEFAULT 12)
RETURNS SETOF jsonb
LANGUAGE sql STABLE
AS $function$
  select (to_jsonb(s) - 'embedding' - 'sim') || jsonb_build_object('similarity', sim)
  from (
    select p.*, 1 - (p.embedding <=> query_embedding::vector(384)) as sim
    from public.projects p
    where p.embedding is not null
    order by p.embedding <=> query_embedding::vector(384)
    limit match_count
  ) s
  where sim > match_threshold
  order by sim desc
$function$;

CREATE OR REPLACE FUNCTION public.match_document_chunks(
  query_embedding real[],
  match_threshold double precision DEFAULT 0.7,
  match_count integer DEFAULT 12)
RETURNS SETOF jsonb
LANGUAGE sql STABLE
AS $function$
  select (to_jsonb(s) - 'embedding' - 'sim') || jsonb_build_object('similarity', sim)
  from (
    select d.*, 1 - (d.embedding <=> query_embedding::vector(384)) as sim
    from public.document_chunks d
    where d.embedding is not null
    order by d.embedding <=> query_embedding::vector(384)
    limit match_count
  ) s
  where sim > match_threshold
  order by sim desc
$function$;

COMMIT;
