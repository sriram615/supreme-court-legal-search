"""
app.py
======
Production-grade asynchronous FastAPI web server for Indian Supreme Court semantic search
and Legal-BERT contradiction classification.

Target Architecture: Intel Mac (x86_64 CPU)
Enforces:
  - torch.set_num_threads(4)
  - faiss.omp_set_num_threads(1)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ─ OPTIMIZATION HEADERS (MUST BE FIRST)
# ─────────────────────────────────────────────────────────────────────────────
import torch
import faiss
# Configure math libraries for physical cores and prevent deadlocks on macOS
torch.set_num_threads(4)
faiss.omp_set_num_threads(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import custom search engine, validator layer, and utils
from search_engine import LegalSearchEngine
from validator import verify_contradiction_async, VerificationVerdict
from utils import _CITATION_PATTERNS, make_token as _make_token

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SC_Server")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ PYDANTIC API SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="The legal brief snippet or argument text to evaluate.",
        examples=["In Manisha Ravindra Panpatil v. State of Maharashtra, the court deprecated prejudicial treatment of female Sarpanchs."],
    )

class SemanticMatch(BaseModel):
    chunk_id: str
    parent_judgment_id: str
    court: str
    tokens_count: int
    text_chunk: str
    extracted_citations: List[str]
    cosine_similarity: float
    rrf_score: Optional[float] = None
    bm25_score: Optional[float] = None


class AnalyzeResponse(BaseModel):
    resolved_citations: Dict[str, str] = Field(..., description="Extracts and resolves canonical case citations using citation_map.json.")
    semantic_matches: List[SemanticMatch] = Field(..., description="Top 3 nearest semantic precedents from the HNSW index.")
    verification_verdict: Optional[VerificationVerdict] = Field(None, description="Referee model verification results.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ LIFESPAN STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Startup: Load Semantic Search Index ──
    logger.info("Initializing search engine state...")
    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")
    
    # Require a pre-built full-corpus index. Never fall back to a limited build.
    if not engine.index_path.exists() or not engine.metadata_path.exists():
        raise FileNotFoundError(
            "Full-corpus FAISS index not found. "
            "Build it first by running:\n\n"
            "  .venv311/bin/python search_engine.py\n\n"
            "This will embed all chunks in aws_court_chunks.jsonl and write:\n"
            "  LAWdata_Corpus_2024/legal_precedents.index\n"
            "  LAWdata_Corpus_2024/metadata_map.json\n"
            "  LAWdata_Corpus_2024/embeddings.npy\n"
        )
    engine.load_index()
    
    app.state.search_engine = engine

    # ── 2. Load Citation Mapping Dictionary ──
    citation_map_path = Path("./LAWdata_Corpus_2024/citation_map.json")
    if citation_map_path.exists():
        logger.info("Loading citation map database: %s", citation_map_path)
        with open(citation_map_path, "r", encoding="utf-8") as f:
            app.state.citation_map = json.load(f)
    else:
        logger.warning("citation_map.json not found. Case resolving will return 'Unknown Case'.")
        app.state.citation_map = {}

    logger.info("Search engine state warm-loaded. Ready for requests.")
    yield
    
    # ── 3. Shutdown Cleanup ──
    logger.info("Cleaning engine cache allocations...")
    del app.state.search_engine
    del app.state.citation_map

# Initialize FastAPI App with lifespan state manager
app = FastAPI(
    title="Indian Supreme Court Legal Analytics Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Legal Brief for citations and semantic matches",
)
async def analyze_legal_brief(request: AnalyzeRequest):
    """
    Run semantic and citation analysis on brief snippets:
    - **Retrieval:** Fetch Top-3 matching Supreme Court precedents using FAISS HNSW.
    - **Citations:** Parse, normalize, and resolve citations from `citation_map.json`.
    """
    try:
        t_start = time.perf_counter()
        
        # ── Stage A: Semantic Precedent Retrieval (Top-3) ──
        engine: LegalSearchEngine = app.state.search_engine
        semantic_matches = engine.search(query=request.text, top_k=3)

        # ── Stage B: Citation Resolution & Cross-Referencing ──
        resolved_citations = {}
        citation_map = app.state.citation_map
        
        for family, pattern in _CITATION_PATTERNS.items():
            matches = pattern.findall(request.text)
            for match in matches:
                # Capture nested regex groups if present
                raw_match = match[0] if isinstance(match, tuple) else match
                raw_match = raw_match.strip()
                token = _make_token(raw_match)
                
                # Cross-reference with our dataset index
                resolved_value = citation_map.get(token, "Unknown Case/Unmapped Citation")
                resolved_citations[raw_match] = f"[{token}] -> {resolved_value}"

        # ── Stage C: Referee LLM Governance Validation ──
        precedent_text = ""
        if semantic_matches:
            precedent_text = semantic_matches[0].get("text_chunk", "")

        verification_verdict = None
        if precedent_text:
            verification_verdict = await verify_contradiction_async(request.text, precedent_text)

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info("API Analyze completed in %.2f ms", latency_ms)

        return AnalyzeResponse(
            resolved_citations=resolved_citations,
            semantic_matches=semantic_matches,
            verification_verdict=verification_verdict,
        )

    except Exception as exc:
        logger.error("Analyze request failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal semantic query execution error: {str(exc)}"
        )

@app.get(
    "/api/v1/health",
    status_code=status.HTTP_200_OK,
    summary="Health and Engine Readiness Check",
)
async def health_check():
    """
    Verify engine components are initialized and check the FAISS index database size.
    """
    try:
        engine: LegalSearchEngine = app.state.search_engine
        is_ready = engine.index is not None and len(app.state.citation_map) > 0
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy" if is_ready else "degraded",
                "engine_ready": is_ready,
                "index_vectors_count": engine.index.ntotal if engine.index else 0,
                "citations_count": len(app.state.citation_map),
                "device": "cpu",
                "thread_tuning": {
                    "torch_threads": torch.get_num_threads(),
                }
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "detail": str(exc)
            }
        )

# Mount static frontend directory right before main entrypoint (after all API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ─ ENTRYPOINT & EXECUTION PATTERN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local debug server execution via uvicorn...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

