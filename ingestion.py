import json
import re
import pickle
import os
from dateutil import parser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

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
    with open('shl_product_catalog.json', 'r', encoding='utf-8') as f:
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

    print("Building BM25 indices...")
    tokenized_chunks = [re.split(r"\W+", c) for c in chunks]
    bm25_chunks = BM25Okapi(tokenized_chunks)
    
    tokenized_names = [re.split(r"\W+", n) for n in names]
    bm25_names = BM25Okapi(tokenized_names)
    
    with open('bm25_indices.pkl', 'wb') as f:
        pickle.dump({
            "bm25_chunks": bm25_chunks,
            "bm25_names": bm25_names,
            "slugs": slugs_list
        }, f)
        
    with open('catalog.json', 'w', encoding='utf-8') as f:
        json.dump(normalized_catalog, f, ensure_ascii=False, indent=2)

    print("Generating Embeddings...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    print("Upserting to Pinecone...")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("PINECONE_API_KEY not set. Skipping Pinecone upsert.")
        return

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "shl-assessments"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, # all-MiniLM-L6-v2 dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(index_name)
    
    batch_size = 100
    vectors = []
    
    for i, slug in enumerate(slugs_list):
        metadata = normalized_catalog[slug].copy()
        
        # pinecone metadata only supports string, number, boolean, or list of strings
        # remove None values
        if metadata['duration_minutes'] is None:
            metadata['duration_minutes'] = -1 # default for missing to allow filtering, or just omit
            del metadata['duration_minutes']
        
        vectors.append({
            "id": slug,
            "values": embeddings[i].tolist(),
            "metadata": metadata
        })
        
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []
            
    if vectors:
        index.upsert(vectors=vectors)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
