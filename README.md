# SHL Assessment Recommender System

An intelligent, conversational AI assistant designed to recommend the best SHL assessment products based on user requirements. This system uses a sophisticated LangGraph agentic workflow with LLMs (Gemini and Groq) to parse complex user constraints (roles, skills, seniority levels) and retrieves matching tests from an SHL product catalog using hybrid search (BM25 + pgvector in Supabase).

## Features

- **Conversational AI Agent**: Implements a LangGraph state machine with specialized nodes for intent classification, state extraction, query generation, and response synthesis.
- **Hybrid Retrieval System**: Combines dense vector search and full-text sparse retrieval (PostgreSQL FTS/BM25) to find exact matches for skills, roles, and assessment categories.
- **Dynamic Constraint Tracking**: Remembers context across multi-turn conversations, accumulating requirements like specific tools (e.g., "Java", "Docker"), job levels, and assessment types (e.g., "cognitive", "personality").
- **Canonical Product Injection**: Automatically injects standard SHL batteries based on role and seniority context (e.g., automatically including OPQ32r and Verify G+ for senior hires, or GSA for talent audits).

## Setup Instructions

### 1. Requirements

- Python 3.10+
- Supabase account (for pgvector database)
- Gemini API Key (for embeddings and agent logic)
- Groq API Key (for high-speed response synthesis)

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory based on `.env.sample`:

```env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_KEY="your_supabase_service_role_key"
API_URL="http://localhost:8000"
```

### 3. Database Setup (Supabase)

You must configure your Supabase instance to support pgvector and full-text search:

1. Open the SQL Editor in your Supabase dashboard.
2. Copy and execute the contents of `scripts/supabase_setup.sql`.

### 4. Data Ingestion

Run the ingestion script to process the raw catalog, generate vectors embeddings, build local fallback BM25 indices, and upsert everything to Supabase:

## Running the Application

### Start the FastAPI Server

To start the API server locally:

```bash
python -m uvicorn app.main:app --reload
```

The server will be available at `http://localhost:8000`.

## Architecture

1. **Gatekeeper**: Classifies intent (VAGUE, CONSTRAINED, REFINE, COMPARE) to guide the workflow.
2. **Role Classifier**: Expands user context into multiple diverse semantic search queries.
3. **State Extractor**: Extracts distinct requirements (skills, duration, remote) and updates persistent state.
4. **Retriever**: Executes multi-query searches against Supabase (pgvector + FTS), applies Reciprocal Rank Fusion, and handles fallback logic.
5. **Synthesizer**: Constructs the final recommendations applying specific SHL battery guidelines.
6. **Schema Validator**: Uses Pydantic to ensure all API outputs match the expected schemas and catalog slugs.
