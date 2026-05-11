import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import pickle
import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from supabase import create_client, Client
from models import ChatRequest, ChatResponse, ConstraintState, Message, ExtractorResponse, RoleExpansion
from llms import call_gemini, call_synthesizer, embed_text

# Global state
catalog_dict = {}
bm25_indices = {}
supabase_client: Client = None

# --- CANONICAL PRODUCT MAPPINGS ---
# These fix the slug aliasing problem (e.g., verify-g vs shl-verify-interactive-g)
CANONICAL_PRODUCTS = {
    "personality": "occupational-personality-questionnaire-opq32r",
    "cognitive": "shl-verify-interactive-g",
    "graduate_sjt": "graduate-scenarios",
    "management_sjt": "management-scenarios",
    "safety": "dependability-and-safety-instrument-dsi",
    "safety_focus": "safety-and-dependability-focus-8-0",
    "gsa": "global-skills-assessment",
    "gsa_dev": "global-skills-development-report",
    "opq_leadership": "opq-leadership-report",
    "opq_ucf": "opq-universal-competency-report-2-0",
    "opq_sales": "opq-mq-sales-report",
    "sales_transform": "salestransformationreport2-0-individualcontributor",
}

# Maps user-mentioned skills/technologies to exact catalog slugs
SKILL_TO_SLUG = {
    "java": "core-java-advanced-level-new",
    "core java": "core-java-advanced-level-new",
    "spring": "spring-new",
    "sql": "sql-new",
    "aws": "amazon-web-services-aws-development-new",
    "amazon web services": "amazon-web-services-aws-development-new",
    "docker": "docker-new",
    "rest": "restful-web-services-new",
    "restful": "restful-web-services-new",
    "angular": "angular-6-new",
    "excel": "microsoft-excel-365-new",
    "ms excel": "microsoft-excel-365-new",
    "word": "microsoft-word-365-new",
    "ms word": "microsoft-word-365-new",
    "hipaa": "hipaa-security",
    "python": "python-new",
    ".net": "dot-net-framework-4-5",
    "c#": "csharp-new",
    "javascript": "javascript-new",
    "react": "react-new",
    "linux": "linux-programming-general",
    "networking": "networking-and-implementation-new",
    "financial accounting": "financial-accounting-new",
    "accounting": "financial-accounting-new",
    "statistics": "basic-statistics-new",
    "medical terminology": "medical-terminology-new",
    "salesforce": "salesforce-development-new",
    "sap": "sap-sd-sales-and-distribution-new",
    "html": "html-css-new",
    "css": "html-css-new",
    "php": "php-new",
    "ruby": "ruby-on-rails-new",
    "data science": "data-science-new",
    "machine learning": "machine-learning-new",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global catalog_dict, bm25_indices, supabase_client
    
    # Load catalog
    if os.path.exists("catalog.json"):
        with open("catalog.json", "r", encoding="utf-8") as f:
            catalog_dict = json.load(f)
            
    # Load BM25 (fallback)
    if os.path.exists("bm25_indices.pkl"):
        try:
            with open("bm25_indices.pkl", "rb") as f:
                bm25_indices = pickle.load(f)
        except Exception:
            pass
            
    # Connect Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        
    yield

app = FastAPI(lifespan=lifespan)

# --- LANGGRAPH STATE ---
class GraphState(TypedDict):
    messages: List[Dict[str, str]]
    turn_count: int
    force_recommend: bool
    intent: str
    constraint_state: dict
    classified_roles: List[str]
    retrieval_queries: List[str]
    retrieval_results: List[dict]
    final_response: dict
    compare_slugs: List[str]

# --- NODES ---

def node_a_turn_budget(state: GraphState):
    print("--- [Node A] Turn Budget ---")
    msgs = [m for m in state["messages"] if m["role"] == "user"]
    turn_count = len(msgs)
    
    force_recommend = False
    if turn_count >= 6:
        force_recommend = True
    
    print(f"Turn count: {turn_count}, Force recommend: {force_recommend}")
    return {"turn_count": turn_count, "force_recommend": force_recommend}

def node_b_gatekeeper(state: GraphState):
    print("--- [Node B] Gatekeeper ---")
    latest_user_msg = state["messages"][-1]["content"]
    conversation_so_far = json.dumps(state["messages"][-6:])  # last few turns for context
    
    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["VAGUE", "CONSTRAINED", "REFINE", "OUT_OF_SCOPE", "COMPARE"]
            },
            "compare_slugs": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["intent"]
    }
    
    prompt = f"""You are an intent classifier for an SHL Assessment Recommender chatbot.

Given the conversation history and the latest user message, classify the user intent.

INTENT DEFINITIONS:
- VAGUE: The user wants assessments but has NOT provided enough specifics to make a useful search. Missing critical info like: what role/job, what level/seniority, what type of assessment, what skills to test, what industry. A single vague sentence without role details is VAGUE.
- CONSTRAINED: The user has provided CLEAR, SPECIFIC details about the role, skills, seniority level, or assessment type needed — enough to run a meaningful search. The message mentions concrete job titles, specific technologies, specific assessment categories (cognitive, personality, SJT), or detailed job descriptions.
- REFINE: The user is ADDING or CHANGING constraints to an existing search in a multi-turn conversation. They are answering a clarifying question or modifying previous requirements. This includes short answers like "English", "US", "Backend-leaning", "Yes", "Drop X", "Add Y" that respond to a previous agent question.
- COMPARE: The user explicitly asks to compare two or more specific named assessments.
- OUT_OF_SCOPE: The user asks about something unrelated to SHL assessments (legal advice, pricing, etc.).

IMPORTANT RULES:
- If this is NOT the first user message and the user is answering a clarifying question, classify as REFINE.
- On the FIRST message, only classify as CONSTRAINED if the user provides MULTIPLE specific details (role + skills + level, or a detailed job description). A single sentence like "We need a solution for senior leadership" is VAGUE.
- Short confirmations like "Yes", "That works", "Perfect", "Go ahead" in a multi-turn context should be REFINE.

EXAMPLES:
- "We need a solution for senior leadership" → VAGUE (no specific role, skills, or assessment type)
- "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?" → CONSTRAINED (specific role, skill, domain)
- "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus." → CONSTRAINED (specific role, level, focus area)
- "English" (answering "what language?") → REFINE
- "Backend-leaning" (answering "what focus?") → REFINE
- "Yes, go ahead" → REFINE
- "Drop REST and add AWS" → REFINE
- "Are we legally required under HIPAA?" → OUT_OF_SCOPE

Conversation History: {conversation_so_far}
Latest User Message: {latest_user_msg}
"""
    
    response = call_gemini("gemini-3-flash-preview", [{"role": "user", "content": prompt}], response_schema=schema)
    try:
        data = json.loads(response)
        print(f"Detected intent: {data.get('intent')}")
        return {
            "intent": data.get("intent", "VAGUE"),
            "compare_slugs": data.get("compare_slugs", [])
        }
    except Exception as e:
        print(f"Gatekeeper error: {e}")
        return {"intent": "VAGUE"}

def node_c_role_classifier(state: GraphState):
    print("--- [Node C] Role Classifier ---")
    # Build full context from all messages
    all_user_msgs = " ".join([m["content"] for m in state["messages"] if m["role"] == "user"])
    c_state = state.get("constraint_state", {})
    
    prompt = f"""You are an expert in SHL assessment products. Based on the user's conversation, generate search queries to find the RIGHT assessments from SHL's catalog.

User's full context: {all_user_msgs}
Current constraints: {json.dumps(c_state)}

Generate 3-5 specific, diverse search queries that will help find ALL relevant assessments. Think about:
1. Role-specific knowledge tests (e.g., "Java programming assessment", "financial accounting test")
2. Personality and behavioral assessments (e.g., "OPQ personality questionnaire workplace behavior")
3. Cognitive/aptitude tests (e.g., "verify numerical reasoning aptitude test")
4. Situational judgment tests (e.g., "graduate scenarios situational judgment")
5. Industry/domain-specific assessments (e.g., "safety dependability manufacturing", "contact center simulation")
6. Skills assessments (e.g., "Excel Word office skills test")

Each query should target a DIFFERENT type of assessment that might be relevant.
Return as a list of role/query strings.
"""
    
    schema = RoleExpansion.model_json_schema()
    response = call_gemini("gemini-3-flash-preview", [{"role": "user", "content": prompt}], response_schema=schema)
    
    try:
        data = json.loads(response)
        roles = data.get("roles", [])
        if not roles: roles = ["General assessment"]
        print(f"Generated queries: {roles}")
        return {"classified_roles": roles}
    except Exception as e:
        print(f"Role Classifier error: {e}")
        return {"classified_roles": ["General assessment"]}

def node_d_state_extractor(state: GraphState):
    print("--- [Node D] State Extractor ---")
    current_state = state.get("constraint_state", ConstraintState().model_dump())
    # Use ALL user messages for complete context
    all_user_msgs = " | ".join([m["content"] for m in state["messages"] if m["role"] == "user"])
    
    prompt = f"""Extract and accumulate ALL constraints from the user's messages into a structured state.

IMPORTANT: Accumulate information from ALL messages, don't just look at the latest one. 
If the user mentioned "Java developer" earlier and now says "Mid-level", combine both into the state.

All User Messages: {all_user_msgs}
Current State: {json.dumps(current_state)}

Extract these fields:
- role: The job role being hired for (e.g., "Java developer", "contact centre agent", "financial analyst")
- seniority: Level (e.g., "senior", "entry-level", "mid-level", "graduate")  
- skills: Specific skills, technologies, or competencies mentioned. Extract EVERY named technology, tool, or skill (e.g., ["Java", "Spring", "SQL", "AWS", "Docker", "REST", "Angular", "Excel", "Word", "HIPAA"])
- labels: Assessment types or categories requested (e.g., ["cognitive", "personality", "SJT", "knowledge", "simulation"])
- language: Required language for assessments
- duration_max: Maximum duration in minutes if mentioned
- remote_required: Whether remote/online is required

IMPORTANT for skills extraction:
- Extract ALL specific technologies, tools, frameworks, and compliance terms mentioned
- "full-stack Java/Spring/SQL" → ["Java", "Spring", "SQL"]
- "Add AWS and Docker. Drop REST" → add AWS, Docker; note REST removal
- "HIPAA compliance, patient records, Spanish" → ["HIPAA", "patient records", "Spanish"]
- "Excel and Word" → ["Excel", "Word"]

If the user explicitly says they don't care about a field, add it to explicit_nulls_to_add.
"""
    
    schema = ExtractorResponse.model_json_schema()
    response = call_gemini("gemini-3-flash-preview", [{"role": "user", "content": prompt}], response_schema=schema)
    
    try:
        data = json.loads(response)
        for k, v in data.items():
            if k == "explicit_nulls_to_add":
                current_state["explicit_nulls"] = list(set(current_state.get("explicit_nulls", []) + v))
            elif v is not None and v != [] and v != "":
                current_state[k] = v
        print(f"Updated Constraint State: {current_state}")
    except Exception as e:
        print(f"State Extractor error: {e}")
        pass
        
    return {"constraint_state": current_state}

def node_e_retriever(state: GraphState):
    print("--- [Node E] Retriever ---")
    intent = state.get("intent")
    
    if intent == "COMPARE":
        slugs = state.get("compare_slugs", [])
        print(f"Comparing slugs: {slugs}")
        results = []
        for slug in slugs:
            query = re.split(r"\W+", slug.lower())
            if "bm25_names" in bm25_indices:
                scores = bm25_indices["bm25_names"].get_scores(query)
                best_idx = max(range(len(scores)), key=scores.__getitem__)
                best_slug = bm25_indices["slugs"][best_idx]
                results.append(catalog_dict.get(best_slug))
        return {"retrieval_results": results}
        
    # Hybrid Multi-Query Retriever
    roles = state.get("classified_roles", ["General assessment"])
    c_state = state.get("constraint_state", {})
    
    # Build comprehensive query list
    queries = []
    
    # 1. Use the classified roles/queries from Node C
    for role in roles:
        query_str = f"{role}"
        if c_state.get("seniority"): query_str += f" {c_state['seniority']}"
        if c_state.get("skills"): query_str += f" {' '.join(c_state['skills'])}"
        queries.append(query_str)
    
    # 2. Add raw user message as a query (captures exact terms the user used)
    all_user_text = " ".join([m["content"] for m in state["messages"] if m["role"] == "user"])
    queries.append(all_user_text)
    
    # 3. Add specific assessment type queries based on constraints
    if c_state.get("skills"):
        for skill in c_state["skills"]:
            queries.append(f"{skill} assessment test")
    
    # 4. Always search for core SHL assessment categories
    if c_state.get("role") or c_state.get("seniority"):
        role_str = c_state.get("role", "")
        seniority_str = c_state.get("seniority", "")
        queries.append(f"personality questionnaire OPQ workplace behavior {role_str}")
        queries.append(f"cognitive aptitude reasoning verify {seniority_str}")
    
    print(f"Running {len(queries)} retrieval queries")
    
    all_results = {}
    
    for query_str in queries:
        dense_results = []
        bm25_results = []
        
        # Try Supabase first
        if supabase_client:
            try:
                # 1. Dense retrieval via pgvector
                emb = embed_text([query_str])[0]
                rpc_args = {"query_embedding": emb, "match_count": 25}
                if c_state.get("duration_max"):
                    rpc_args["filter_duration"] = c_state["duration_max"]
                if c_state.get("remote_required"):
                    rpc_args["filter_remote"] = True
                    
                res = supabase_client.rpc("match_assessments", rpc_args).execute()
                for item in res.data:
                    dense_results.append(item["slug"])
                
                # 2. Sparse retrieval via pg_trgm/FTS
                res_fts = supabase_client.rpc("search_assessments_fts", {"query_text": query_str, "match_count": 25}).execute()
                for item in res_fts.data:
                    bm25_results.append(item["slug"])
            except Exception as e:
                print(f"Supabase search error: {e}")
        
        # Local BM25 fallback if FTS failed/missing
        if not bm25_results and "bm25_chunks" in bm25_indices:
            q_tokens = re.split(r"\W+", query_str.lower())
            scores = bm25_indices["bm25_chunks"].get_scores(q_tokens)
            top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:25]
            for i in top_k:
                bm25_results.append(bm25_indices["slugs"][i])
                
        # RRF Merge
        for rank, slug in enumerate(dense_results):
            all_results[slug] = all_results.get(slug, 0) + 1.0 / (60 + rank)
        for rank, slug in enumerate(bm25_results):
            all_results[slug] = all_results.get(slug, 0) + 1.0 / (60 + rank)
    
    # 5. Direct keyword matching on catalog — catch exact skill/tool names
    user_text_lower = all_user_text.lower()
    for slug, item in catalog_dict.items():
        name_lower = item.get("name", "").lower()
        desc_lower = item.get("description", "").lower()
        
        # Boost items whose names directly contain user-mentioned skills
        if c_state.get("skills"):
            for skill in c_state["skills"]:
                skill_lower = skill.lower()
                if skill_lower in name_lower:
                    all_results[slug] = all_results.get(slug, 0) + 0.1
                if skill_lower in desc_lower:
                    all_results[slug] = all_results.get(slug, 0) + 0.03
        
        # Boost items that match user text keywords
        user_keywords = set(re.split(r"\W+", user_text_lower))
        name_words = set(re.split(r"\W+", name_lower))
        overlap = user_keywords & name_words
        if len(overlap) >= 2:
            all_results[slug] = all_results.get(slug, 0) + 0.05 * len(overlap)
            
    top_slugs = sorted(all_results.keys(), key=lambda k: all_results[k], reverse=True)[:12]
    
    # --- FORCE-INJECT CANONICAL PRODUCTS ---
    injected = set()
    
    # Always inject OPQ32r for any role-based query
    if c_state.get("role") or c_state.get("skills"):
        injected.add(CANONICAL_PRODUCTS["personality"])
        injected.add(CANONICAL_PRODUCTS["cognitive"])
    
    # Seniority-based injections
    seniority = (c_state.get("seniority") or "").lower()
    if seniority in ["graduate", "entry-level", "entry level"]:
        injected.add(CANONICAL_PRODUCTS["graduate_sjt"])
    if seniority in ["senior", "executive", "director", "cxo"]:
        injected.add(CANONICAL_PRODUCTS["opq_leadership"])
    
    # Purpose-based injections
    all_user_lower = all_user_text.lower()
    if any(kw in all_user_lower for kw in ["re-skill", "reskill", "development", "talent audit", "restructuring"]):
        injected.add(CANONICAL_PRODUCTS["gsa"])
        injected.add(CANONICAL_PRODUCTS["gsa_dev"])
    
    # Domain-based injections
    if any(kw in all_user_lower for kw in ["safety", "plant", "manufacturing", "chemical", "hazard"]):
        injected.add(CANONICAL_PRODUCTS["safety"])
    if any(kw in all_user_lower for kw in ["sales", "selling", "account management"]):
        injected.add(CANONICAL_PRODUCTS["opq_sales"])
        injected.add(CANONICAL_PRODUCTS["sales_transform"])
    if any(kw in all_user_lower for kw in ["leadership", "leader", "cxo", "director"]):
        injected.add(CANONICAL_PRODUCTS["opq_leadership"])
        injected.add(CANONICAL_PRODUCTS["opq_ucf"])
    
    # Skill-to-slug direct mapping
    if c_state.get("skills"):
        for skill in c_state["skills"]:
            skill_key = skill.lower().strip()
            if skill_key in SKILL_TO_SLUG:
                injected.add(SKILL_TO_SLUG[skill_key])
    
    # Add injected slugs that aren't already in top results
    for slug in injected:
        if slug not in top_slugs and slug in catalog_dict:
            top_slugs.append(slug)
    
    final_results = [catalog_dict[slug] for slug in top_slugs if slug in catalog_dict]
    print(f"Retrieved {len(final_results)} results (injected {len(injected)} canonical).")
    
    return {"retrieval_results": final_results}

def node_f_synthesizer(state: GraphState):
    print("--- [Node F] Synthesizer ---")
    history = json.dumps(state["messages"])
    force_rec = state.get("force_recommend", False)
    intent = state.get("intent", "VAGUE")
    c_state = state.get("constraint_state", {})
    
    # Build a compact retrieval context with exact slugs
    retrieval_items = state.get("retrieval_results", [])
    retrieval_block = ""
    for i, item in enumerate(retrieval_items, 1):
        slug = item.get("slug", "")
        name = item.get("name", "")
        dur = item.get("duration_minutes")
        dur_str = f"{dur}min" if dur else "untimed"
        keys = ", ".join(item.get("keys", []))
        
        retrieval_block += f"{i}. {name} | slug={slug} | {keys} | {dur_str}\n"
    
    # Determine what info is missing for smarter VAGUE questions
    missing_dims = []
    if not c_state.get("role"): missing_dims.append("role")
    if not c_state.get("seniority"): missing_dims.append("seniority")
    if not c_state.get("skills"): missing_dims.append("skills")
    
    # Check if purpose (hiring vs development) is clear
    all_msgs = " ".join([m["content"] for m in state["messages"] if m["role"] == "user"]).lower()
    purpose_clear = any(kw in all_msgs for kw in ["hiring", "selection", "screening", "recruit", "development", "reskill", "re-skill", "audit", "training"])
    if not purpose_clear: missing_dims.append("purpose")
    
    missing_info = ", ".join(missing_dims) if missing_dims else "none"
    
    prompt = f"""You are an expert SHL Assessment Recommender. Your job is to recommend the best SHL assessment products for a hiring or development scenario.

## FEW-SHOT EXAMPLES (learn the correct battery patterns from these):

EXAMPLE 1 — Tech Hiring (Senior Java Engineer):
User: "I'm hiring a senior full-stack engineer. Core Java, Spring, SQL on the backend, Angular on the front. Backend-leaning."
Correct Battery: Core Java Advanced + Spring + SQL + Angular 6 + OPQ32r + Verify G+ (6-7 items)
Key: Map EACH named technology to its specific test. Always add OPQ32r + Verify G+.

EXAMPLE 2 — Sales Re-skilling:
User: "We need to re-skill our Sales organization as part of a talent audit."
Correct Battery: Global Skills Assessment + Global Skills Development Report + OPQ32r + OPQ MQ Sales Report + Sales Transformation 2.0 (5 items)
Key: Development/reskilling uses GSA + GSA Dev Report, NOT Verify G+. Sales roles get OPQ MQ Sales Report.

EXAMPLE 3 — Senior Leadership Selection:
User: "CXOs, director-level, 15+ years experience. Selection against leadership benchmark."
Correct Battery: OPQ32r + OPQ Leadership Report + OPQ Universal Competency Report 2.0 (3 items)
Key: Executive selection is personality-focused. No knowledge tests needed.

EXAMPLE 4 — Safety-Critical Roles:
User: "Plant operators at a chemical facility. Safety and reliability are paramount."
Correct Battery: DSI + Safety & Dependability Focus + Workplace Health & Safety + domain knowledge test (4-7 items)
Key: Safety roles ALWAYS get DSI.

EXAMPLE 5 — Graduate Financial Analyst:
User: "Graduate-level financial analysts. Need numerical reasoning, finance knowledge, and SJT."
Correct Battery: Verify Numerical Reasoning + Financial Accounting + Basic Statistics + Graduate Scenarios + OPQ32r (5+ items)
Key: Graduates get Graduate Scenarios. Map specific knowledge areas to specific tests.

EXAMPLE 6 — Contact Centre Screening:
User: "500 entry-level contact centre agents, inbound calls, English US."
Correct Battery: Contact Center Call Simulation + Customer Service Phone Simulation + Entry Level Customer Service + SVAR Spoken English US (4-5 items)
Key: High-volume screening uses Solutions and Simulations.

EXAMPLE 7 — Healthcare Admin:
User: "Healthcare admin staff, bilingual Spanish, HIPAA compliance, patient records, Word."
Correct Battery: HIPAA Security + Medical Terminology + Microsoft Word 365 Essentials + OPQ32r + DSI (5 items)
Key: Map EACH specific compliance/tool to its test. Healthcare gets DSI for patient safety.

## RULES
- When recommending, include ALL relevant items from the retrieval list — err on the side of MORE, not fewer
- For hiring: ALWAYS include OPQ32r (personality) AND Verify G+ (cognitive) unless the role is executive-only
- For development/reskilling: Include GSA + GSA Dev Report instead of Verify G+
- Map EVERY named skill/technology to its specific knowledge test if available in the list
- Use EXACT slug from the retrieval list to construct URL: https://www.shl.com/products/product-catalog/view/SLUG/

HISTORY: {history}
CONSTRAINTS: {json.dumps(c_state)}
MISSING INFO: {missing_info}

AVAILABLE ASSESSMENTS (use ONLY these, URL = https://www.shl.com/products/product-catalog/view/SLUG/):
{retrieval_block}
INTENT: {intent} | FORCE: {force_rec}

Respond with JSON: {{"reply": str, "recommendations": [{{"name": exact_name, "url": "https://www.shl.com/products/product-catalog/view/SLUG/", "test_type": keys, "duration": dur, "rationale": why}}], "end_of_conversation": bool}}

## DECISION LOGIC
If VAGUE and not FORCE:
  Ask ONE targeted clarifying question about the most important missing info ({missing_info}).
  Use concrete examples from SHL's catalog to help the user decide, e.g.:
  - Missing role: "What specific role are you hiring for? E.g., software engineer, financial analyst, sales rep, contact centre agent..."
  - Missing seniority: "What seniority level? Graduate/entry-level gets Graduate Scenarios; senior gets advanced tests; executive gets leadership reports."
  - Missing purpose: "Is this for hiring/selection, or for development/re-skilling? This changes which products I recommend."
  - Missing skills: "What specific skills or technologies should we assess? E.g., Java, SQL, Excel, HIPAA..."
  Return empty recommendations. Set end_of_conversation=false.

If CONSTRAINED/REFINE/FORCE:
  Build a COMPREHENSIVE battery of 3-10 assessments following the few-shot patterns above.
  Include EVERY matching item from the retrieval list.
  Set end_of_conversation=true.

If COMPARE:
  Compare the named assessments. Include them as recommendations too. Set end_of_conversation=false.

If OUT_OF_SCOPE: decline. Empty recommendations.
"""
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        resp_str = call_synthesizer(messages)
        
        # Clean up markdown if present
        json_match = re.search(r"```json\s*(.*?)\s*```", resp_str, re.DOTALL)
        if json_match:
            resp_str = json_match.group(1)
        else:
            # Try to find the first '{' and last '}'
            start = resp_str.find('{')
            end = resp_str.rfind('}')
            if start != -1 and end != -1:
                resp_str = resp_str[start:end+1]

        final = json.loads(resp_str)
        return {"final_response": final}
    except Exception as e:
        print(f"Synthesizer error: {e}")
        return {"final_response": {"reply": "Sorry, I had an error processing that.", "recommendations": [], "end_of_conversation": False}}

def node_g_schema_validator(state: GraphState):
    print("--- [Node G] Schema Validator ---")
    resp = state.get("final_response", {})
    
    # Pre-process: normalize recommendation fields before Pydantic
    if "recommendations" in resp and isinstance(resp["recommendations"], list):
        for rec in resp["recommendations"]:
            if isinstance(rec, dict):
                # Convert list fields to comma-joined strings
                if isinstance(rec.get("test_type"), list):
                    rec["test_type"] = ", ".join(str(x) for x in rec["test_type"])
                if isinstance(rec.get("duration"), list):
                    rec["duration"] = ", ".join(str(x) for x in rec["duration"])
                # Ensure all required fields exist
                rec.setdefault("test_type", "Unknown")
                rec.setdefault("duration", "Unknown")
                rec.setdefault("rationale", "")
                rec.setdefault("name", "Unknown")
                rec.setdefault("url", "")
    
    try:
        valid_resp = ChatResponse(**resp)
        
        # Validate URLs against catalog
        valid_recs = []
        for rec in valid_resp.recommendations:
            # Extract slug from URL
            parts = rec.url.rstrip('/').split('/')
            slug = parts[-1] if parts else ""
            
            if slug in catalog_dict:
                # Ensure URL format is correct
                rec.url = f"https://www.shl.com/products/product-catalog/view/{slug}/"
                valid_recs.append(rec)
            else:
                # Try case-insensitive slug match
                matched_slug = None
                slug_lower = slug.lower()
                for s in catalog_dict:
                    if s.lower() == slug_lower:
                        matched_slug = s
                        break
                
                if not matched_slug:
                    # Try name-based lookup (case-insensitive)
                    rec_name_lower = rec.name.lower().strip()
                    for s, item in catalog_dict.items():
                        if item.get("name", "").lower().strip() == rec_name_lower:
                            matched_slug = s
                            break
                
                if not matched_slug:
                    # Try substring match on name
                    rec_name_lower = rec.name.lower().strip()
                    for s, item in catalog_dict.items():
                        item_name_lower = item.get("name", "").lower().strip()
                        if rec_name_lower in item_name_lower or item_name_lower in rec_name_lower:
                            matched_slug = s
                            break
                
                if matched_slug:
                    rec.url = f"https://www.shl.com/products/product-catalog/view/{matched_slug}/"
                    valid_recs.append(rec)
                    print(f"  Matched '{rec.name}' via fallback to slug '{matched_slug}'")
                else:
                    print(f"  WARNING: Could not match '{rec.name}' (slug='{slug}') to catalog")
                
        valid_resp.recommendations = valid_recs
        
        # Safety: never end conversation with 0 recommendations unless it's explicitly intended
        if valid_resp.end_of_conversation and len(valid_recs) == 0:
            intent = state.get("intent", "")
            if intent not in ["OUT_OF_SCOPE"]:
                valid_resp.end_of_conversation = False
                print("  Safety: Reset end_of_conversation to False (0 recommendations)")
        
        print(f"Validation success. Recommendations kept: {len(valid_recs)}")
        return {"final_response": valid_resp.model_dump()}
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")

        return {"final_response": {"reply": "Validation error.", "recommendations": [], "end_of_conversation": False}}

# --- GRAPH DEFINITION ---
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("node_a", node_a_turn_budget)
    workflow.add_node("node_b", node_b_gatekeeper)
    workflow.add_node("node_c", node_c_role_classifier)
    workflow.add_node("node_d", node_d_state_extractor)
    workflow.add_node("node_e", node_e_retriever)
    workflow.add_node("node_f", node_f_synthesizer)
    workflow.add_node("node_g", node_g_schema_validator)
    
    workflow.set_entry_point("node_a")
    
    def router_a(state):
        if state.get("turn_count", 0) >= 8:
            state["intent"] = "FORCE_CLOSE"
            state["force_recommend"] = True
            return "node_f"
        return "node_b"
        
    workflow.add_conditional_edges("node_a", router_a)
    
    def router_b(state):
        intent = state.get("intent", "VAGUE")
        if intent == "VAGUE":
            # For VAGUE, still run retrieval so we have results ready
            return "node_c"
        elif intent == "OUT_OF_SCOPE":
            return "node_f"
        elif intent == "COMPARE":
            return "node_e"
        else: # CONSTRAINED, REFINE
            return "node_c"
            
    workflow.add_conditional_edges("node_b", router_b)
    
    workflow.add_edge("node_c", "node_d")
    workflow.add_edge("node_d", "node_e")
    workflow.add_edge("node_e", "node_f")
    workflow.add_edge("node_f", "node_g")
    workflow.add_edge("node_g", END)
    
    return workflow.compile()

graph = build_graph()

# --- FASTAPI ROUTES ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    msgs = [m.model_dump() for m in request.messages]
    initial_state = {
        "messages": msgs,
        "constraint_state": {},
        "classified_roles": [],
        "retrieval_queries": [],
        "retrieval_results": [],
        "compare_slugs": []
    }
    
    result = graph.invoke(initial_state)
    return result.get("final_response", {"reply": "Error", "recommendations": [], "end_of_conversation": False})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
