-- Speed up nearest-name search. Safe to run more than once.
-- Build these after the embedding columns are filled.

CREATE INDEX IF NOT EXISTS propai_places_embedding_hnsw
  ON public.propai_places USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS propai_raw_embedding_hnsw
  ON public.propai_raw USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS propai_project_status_embedding_hnsw
  ON public.propai_project_status USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS propai_brokers_embedding_hnsw
  ON public.propai_brokers USING hnsw (embedding vector_cosine_ops);
