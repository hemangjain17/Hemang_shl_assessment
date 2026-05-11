import json
import re
import pickle
import os
from dateutil import parser
from supabase import create_client, Client
from dotenv import load_dotenv
from app.llms import embed_text

load_dotenv()

def get_slug(url):
    # e.g. https://www.shl.com/products/product-catalog/view/global-skills-development-report/
    parts = url.rstrip('/').split('/')
    if len(parts) > 0:
        return parts[-1]
    return ""

def normalize_duration(duration_raw, duration_field):
    match = re.search(r'(\d+)', duration_field)
    if match: return int(match.group(1))
    match = re.search(r'(\d+)', duration_raw)
    if match: return int(match.group(1))
    return None

def extract_test_types(keys):
    types = []
    if "Knowledge & Skills" in keys: types.append("K")
    if "Personality & Behavior" in keys or "Biodata & Situational Judgment" in keys: types.append("P")
    if "Simulations" in keys: types.append("S")
    if "Competencies" in keys or "Development & 360" in keys: types.append("B")
    if not types: types.append("D")
    return list(set(types))

def main():
    print("Loading raw catalog...")
    with open('data/shl_product_catalog.json', 'r', encoding='utf-8') as f:
        raw_catalog = json.load(f)
        
    print("Deduplicating...")
    deduped = {}
    for item in raw_catalog:
        slug = get_slug(item.get('link', ''))
        if not slug: continue
        
        # Keep latest
        if slug in deduped:
            current_time = parser.parse(deduped[slug]['scraped_at'])
            new_time = parser.parse(item['scraped_at'])
            if new_time > current_time:
                deduped[slug] = item
        else:
            deduped[slug] = item

    print(f"Total unique records: {len(deduped)}")
    
    normalized_catalog = {}
    chunks = []
    names = []
    slugs_list = []
    
    print("Normalizing and constructing chunks...")
    for slug, item in deduped.items():
        dur = normalize_duration(item.get('duration_raw', ''), item.get('duration', ''))
        remote = item.get('remote', '').lower() == 'yes'
        adaptive = item.get('adaptive', '').lower() == 'yes'
        types = extract_test_types(item.get('keys', []))
        
        normalized_record = {
            "entity_id": item.get('entity_id'),
            "name": item.get('name'),
            "link": item.get('link'),
            "duration_minutes": dur,
            "remote": remote,
            "adaptive": adaptive,
            "test_type_codes": types,
            "job_level_codes": item.get('job_levels', []),
            "languages": item.get('languages', []),
            "description": item.get('description', ''),
            "keys": item.get('keys', []),
            "slug": slug
        }
        normalized_catalog[slug] = normalized_record
        
        # Chunk construction
        name_x3 = f"{item.get('name')} {item.get('name')} {item.get('name')}"
        type_str = " ".join(item.get('keys', []))
        flags = []
        if remote: flags.append("remote")
        if adaptive: flags.append("adaptive")
        flags_str = " ".join(flags)
        
        chunk = f"{name_x3} {type_str} {item.get('description', '')} {flags_str}".lower()
        chunks.append(chunk)
        names.append(item.get('name', '').lower())
        slugs_list.append(slug)

    print("Building BM25 indices (for local fallback)...")
    try:
        from rank_bm25 import BM25Okapi
        tokenized_chunks = [re.split(r"\W+", c) for c in chunks]
        bm25_chunks = BM25Okapi(tokenized_chunks)
        
        tokenized_names = [re.split(r"\W+", n) for n in names]
        bm25_names = BM25Okapi(tokenized_names)
        
        with open('data/bm25_indices.pkl', 'wb') as f:
            pickle.dump({
                "bm25_chunks": bm25_chunks,
                "bm25_names": bm25_names,
                "slugs": slugs_list
            }, f)
    except ImportError:
        print("rank_bm25 not installed, skipping local BM25 generation.")
        
    with open('data/catalog.json', 'w', encoding='utf-8') as f:
        json.dump(normalized_catalog, f, ensure_ascii=False, indent=2)

    print("Generating Embeddings (using Gemini API)...")
    # Process in batches to avoid API limits
    batch_size = 50
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        embeddings.extend(embed_text(batch))
    
    print("Upserting to Supabase...")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("Supabase credentials not set. Skipping Supabase upsert.")
        return

    supabase: Client = create_client(supabase_url, supabase_key)
    
    upsert_batch_size = 100
    rows = []
    
    for i, slug in enumerate(slugs_list):
        metadata = normalized_catalog[slug].copy()
        
        # Clean the chunk string so it can be safely cast to tsvector by Postgres
        import re
        safe_search_text = re.sub(r'[^a-z0-9\s]', ' ', chunks[i].lower())
        safe_search_text = re.sub(r'\s+', ' ', safe_search_text).strip()
        
        row = {
            "slug": slug,
            "name": metadata.get("name", ""),
            "description": metadata.get("description", ""),
            "link": metadata.get("link", ""),
            "duration_minutes": metadata.get("duration_minutes"),
            "remote": metadata.get("remote", False),
            "adaptive": metadata.get("adaptive", False),
            "test_type_codes": metadata.get("test_type_codes", []),
            "job_level_codes": metadata.get("job_level_codes", []),
            "languages": metadata.get("languages", []),
            "keys": metadata.get("keys", []),
            "entity_id": metadata.get("entity_id", ""),
            "embedding": embeddings[i],
            "search_text": safe_search_text
        }
        rows.append(row)
        
        if len(rows) >= upsert_batch_size:
            supabase.table("assessments").upsert(rows).execute()
            rows = []
            
    if rows:
        supabase.table("assessments").upsert(rows).execute()
        
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
