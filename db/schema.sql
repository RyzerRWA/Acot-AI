-- ACOT AI data tables for plain PostgreSQL (no pgvector).
-- Embeddings are real[] (384 dims); similarity is computed in SQL.
-- ponytail: brute-force scan, fine for ~thousands of rows. Past ~50k rows, install
-- pgvector, change the column to vector(384) and add an hnsw index.

create or replace function public.cosine_sim(a real[], b real[]) returns double precision
language sql immutable parallel safe as $$
  select sum(x::float8 * y::float8) / nullif(sqrt(sum(x::float8 * x::float8)) * sqrt(sum(y::float8 * y::float8)), 0)
  from unnest(a, b) as t(x, y)
$$;

create table if not exists public.communities (
  id bigserial primary key,
  slug text not null unique,
  name text not null,
  city text not null,
  latitude double precision,
  longitude double precision,
  photos text[] not null default '{}',
  sell_properties_count integer not null default 0,
  rent_properties_count integer not null default 0,
  projects_count integer not null default 0,
  pool_projects_count integer not null default 0,
  total_count integer not null default 0,
  assigned_agents jsonb not null default '[]'::jsonb,
  knowledge_text text not null,
  embedding real[],
  source text,
  fetched_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists communities_city_idx on public.communities (city);
create index if not exists communities_total_count_idx on public.communities (total_count desc);

create table if not exists public.sub_communities (
  id bigserial primary key,
  slug text not null unique,
  name text not null,
  city text not null,
  community text not null,
  community_slug text,
  latitude double precision,
  longitude double precision,
  photos text[] not null default '{}',
  knowledge_text text not null,
  embedding real[],
  source text,
  fetched_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists sub_communities_community_idx on public.sub_communities (community);
create index if not exists sub_communities_community_slug_idx on public.sub_communities (community_slug);
create index if not exists sub_communities_city_idx on public.sub_communities (city);

create table if not exists public.projects (
  id bigint primary key,
  slug text not null unique,
  name text not null,
  description text,
  status text,
  project_status text,
  pool_type text,
  developer_id integer,
  developer_name text,
  developer_logo text,
  city text,
  community text,
  sub_community text,
  latitude double precision,
  longitude double precision,
  price numeric,
  size_min numeric,
  size_max numeric,
  bedroom_min integer,
  bedroom_max integer,
  handover_time timestamptz,
  photos text[] not null default '{}',
  property_types text[] not null default '{}',
  amenities text[] not null default '{}',
  brochure_url text,
  property_finder_url text,
  payload jsonb not null default '{}'::jsonb,
  knowledge_text text not null,
  embedding real[],
  source text,
  fetched_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists projects_city_idx on public.projects (city);
create index if not exists projects_community_idx on public.projects (community);
create index if not exists projects_developer_idx on public.projects (developer_name);
create index if not exists projects_status_idx on public.projects (status);
create index if not exists projects_price_idx on public.projects (price);

create table if not exists public.document_chunks (
  id bigserial primary key,
  document_id text,
  document_title text,
  document_url text,
  document_type text,
  publisher text,
  chunk_id integer,
  content text not null,
  embedding real[],
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Vector search. Same names/params as the old Supabase RPCs; each returns rows as jsonb.
create or replace function public.match_communities(
  query_embedding real[], match_threshold double precision default 0.7, match_count integer default 12
) returns setof jsonb language sql stable as $$
  select jsonb_build_object('id', id, 'slug', slug, 'name', name, 'city', city, 'photos', photos,
                            'total_count', total_count, 'knowledge_text', knowledge_text, 'similarity', sim)
  from (select c.*, public.cosine_sim(c.embedding, query_embedding) as sim
        from public.communities c where c.embedding is not null) s
  where sim > match_threshold order by sim desc limit match_count
$$;

create or replace function public.match_projects(
  query_embedding real[], match_threshold double precision default 0.7, match_count integer default 12
) returns setof jsonb language sql stable as $$
  select (to_jsonb(s) - 'embedding' - 'sim') || jsonb_build_object('similarity', sim)
  from (select p.*, public.cosine_sim(p.embedding, query_embedding) as sim
        from public.projects p where p.embedding is not null) s
  where sim > match_threshold order by sim desc limit match_count
$$;

create or replace function public.match_document_chunks(
  query_embedding real[], match_threshold double precision default 0.7, match_count integer default 12
) returns setof jsonb language sql stable as $$
  select (to_jsonb(s) - 'embedding' - 'sim') || jsonb_build_object('similarity', sim)
  from (select d.*, public.cosine_sim(d.embedding, query_embedding) as sim
        from public.document_chunks d where d.embedding is not null) s
  where sim > match_threshold order by sim desc limit match_count
$$;
