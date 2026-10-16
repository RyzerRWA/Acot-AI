-- =====================================================================
-- 001_pgvector_rollback.sql
-- Undo 001_pgvector.sql: back to real[] columns and cosine_sim() functions.
-- Embedding values are preserved. Runs in one transaction.
-- =====================================================================

BEGIN;

DROP INDEX IF EXISTS public.communities_embedding_hnsw;
DROP INDEX IF EXISTS public.sub_communities_embedding_hnsw;
DROP INDEX IF EXISTS public.projects_embedding_hnsw;
DROP INDEX IF EXISTS public.document_chunks_embedding_hnsw;

ALTER TABLE public.communities     ALTER COLUMN embedding TYPE real[] USING embedding::real[];
ALTER TABLE public.sub_communities ALTER COLUMN embedding TYPE real[] USING embedding::real[];
ALTER TABLE public.projects        ALTER COLUMN embedding TYPE real[] USING embedding::real[];
ALTER TABLE public.document_chunks ALTER COLUMN embedding TYPE real[] USING embedding::real[];

CREATE OR REPLACE FUNCTION public.match_communities(query_embedding real[], match_threshold double precision DEFAULT 0.7, match_count integer DEFAULT 12)
RETURNS SETOF jsonb
LANGUAGE sql STABLE
AS $function$
  select jsonb_build_object('id', id, 'slug', slug, 'name', name, 'city', city, 'photos', photos,
                            'total_count', total_count, 'knowledge_text', knowledge_text, 'similarity', sim)
  from (select c.*, public.cosine_sim(c.embedding, query_embedding) as sim
        from public.communities c where c.embedding is not null) s
  where sim > match_threshold order by sim desc limit match_count
$function$;

CREATE OR REPLACE FUNCTION public.match_projects(query_embedding real[], match_threshold double precision DEFAULT 0.7, match_count integer DEFAULT 12)
RETURNS SETOF jsonb
LANGUAGE sql STABLE
AS $function$
  select (to_jsonb(s) - 'embedding' - 'sim') || jsonb_build_object('similarity', sim)
  from (select p.*, public.cosine_sim(p.embedding, query_embedding) as sim
        from public.projects p where p.embedding is not null) s
  where sim > match_threshold order by sim desc limit match_count
$function$;

CREATE OR REPLACE FUNCTION public.match_document_chunks(query_embedding real[], match_threshold double precision DEFAULT 0.7, match_count integer DEFAULT 12)
RETURNS SETOF jsonb
LANGUAGE sql STABLE
AS $function$
  select (to_jsonb(s) - 'embedding' - 'sim') || jsonb_build_object('similarity', sim)
  from (select d.*, public.cosine_sim(d.embedding, query_embedding) as sim
        from public.document_chunks d where d.embedding is not null) s
  where sim > match_threshold order by sim desc limit match_count
$function$;

COMMIT;
