# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Server (Docker - Recommended)
- **Run container**: `docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag` - Starts the MCP server in container
- **Docker build** (if needed): `docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .`

### Alternative: Direct Python Execution
- **Main server**: `uv run src/crawl4ai_mcp.py` - Starts the MCP server (SSE or stdio transport based on TRANSPORT env var)
- **Install dependencies**: `uv pip install -e .` followed by `crawl4ai-setup`
- **Test embeddings**: `python test_embeddings.py` - Test local BGE embedding functionality

## Architecture Overview

### Core Components

**MCP Server (`src/crawl4ai_mcp.py`)**
- FastMCP-based server providing web crawling and RAG tools via Model Context Protocol
- Manages async lifecycle with crawler, Supabase client, and optional reranking model
- Provides 5 main tools: `crawl_single_page`, `smart_crawl_url`, `get_available_sources`, `perform_rag_query`, `search_code_examples`
- Knowledge graph tools (`parse_github_repository`, `check_ai_script_hallucinations`, `query_knowledge_graph`) are commented out

**Local Embeddings (`src/local_embeddings.py`)**
- BAAI-bge-large-en-v1.5 model integration 
- Generates 1024-dimensional embeddings with proper BGE preprocessing (mean pooling, normalization)
- Loads exclusively from local model path (no downloads), requires BGE_MODEL_PATH environment variable
- Singleton service pattern with batch processing support

**Utilities (`src/utils.py`)**  
- Supabase integration for vector storage and search
- Content chunking with smart markdown splitting
- Code block extraction and contextual embedding processing
- Hybrid search combining vector similarity and keyword matching

**Knowledge Graph System (`knowledge_graphs/`) - COMMENTED OUT**
- Neo4j-based code analysis for AI hallucination detection (not currently used)
- Files exist but functionality is commented out in main MCP server
- Can be re-enabled in future by uncommenting relevant code sections

### Data Flow Architecture

1. **Crawling**: Web content → Crawl4AI → Chunking → Local BGE embeddings → Supabase vector storage
2. **RAG Search**: Query → BGE embedding → Vector similarity + optional keyword search → Optional reranking → Results

### Key Design Patterns

**Conditional Feature Loading**: Tools and functionality enabled via environment flags (USE_RERANKING, USE_AGENTIC_RAG, USE_HYBRID_SEARCH, USE_CONTEXTUAL_EMBEDDINGS). Note: USE_KNOWLEDGE_GRAPH is commented out.

**Lifespan Context Management**: Single async context manager handles all service lifecycles (crawler, database connections, ML models)

**Batch Processing**: Embeddings, crawling, and database operations use concurrent processing with configurable batch sizes

**Fallback Strategies**: Graceful degradation when optional services (LLM features for contextual embeddings/summaries) are unavailable

## Environment Configuration

Required: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `BGE_MODEL_PATH`
Optional: `MODEL_CHOICE` (for LLM features like contextual embeddings)
Transport: `HOST`, `PORT`, `TRANSPORT` (sse/stdio)

## Database Schema

Supabase with pgvector extension:
- `crawled_pages`: Document chunks with embeddings
- `code_examples`: Extracted code blocks with embeddings  
- `sources`: Domain-level metadata and summaries
- Custom RPC functions: `match_crawled_pages`, `match_code_examples`

Neo4j schema: (Commented out - knowledge graph functionality not currently used)

## Local Model Integration

The codebase migrated from OpenAI embeddings to local BAAI-bge-large-en-v1.5 model. When modifying embedding functionality:
- Use `local_embeddings.py` service, not direct transformers calls
- Embeddings are 1024-dimensional (not OpenAI's 1536)
- Model must be pre-installed locally at `BGE_MODEL_PATH` (no automatic downloads)
- GPU acceleration available if CUDA present