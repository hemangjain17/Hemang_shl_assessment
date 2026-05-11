-- Enable extensions
create extension if not exists vector;
create extension if not exists pg_trgm;

-- Main assessments table
drop table if exists assessments cascade;
create table assessments (
  slug text primary key,
  name text not null,
  description text,
  link text,
  duration_minutes int,
  remote boolean default false,
  adaptive boolean default false,
  test_type_codes text[],
  job_level_codes text[],
  languages text[],
  keys text[],
  entity_id text,
  embedding vector(3072),        -- gemini-embedding-2 dimension
  search_text tsvector           -- for FTS
);

-- Indexes
create index if not exists idx_assessments_embedding 
  on assessments using ivfflat (embedding vector_cosine_ops) with (lists = 20);
create index if not exists idx_assessments_search 
  on assessments using gin (search_text);
create index if not exists idx_assessments_name_trgm 
  on assessments using gin (name gin_trgm_ops);

-- RPC: Vector similarity search
create or replace function match_assessments(
  query_embedding vector(3072),
  match_count int default 25,
  filter_duration int default null,
  filter_remote boolean default null
)
returns table (slug text, name text, similarity float)
language plpgsql as $$
begin
  return query
    select a.slug, a.name, 
           1 - (a.embedding <=> query_embedding) as similarity
    from assessments a
    where (filter_duration is null or a.duration_minutes <= filter_duration)
      and (filter_remote is null or a.remote = filter_remote)
    order by a.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- RPC: Full-text search (replaces BM25)
create or replace function search_assessments_fts(
  query_text text,
  match_count int default 25
)
returns table (slug text, name text, rank float)
language plpgsql as $$
begin
  return query
    select a.slug, a.name,
           ts_rank_cd(a.search_text, plainto_tsquery('english', query_text)) as rank
    from assessments a
    where a.search_text @@ plainto_tsquery('english', query_text)
    order by rank desc
    limit match_count;
end;
$$;
