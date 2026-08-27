"""
DEPRECATION NOTICE: main.py is deprecated.
Please use app.py (uvicorn app:app) for production deployments.
app.py features async lifespan management, strict index missing validation,
and multi-stage intent verification.
"""

import time
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from search_engine import LegalSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SearchAPI")
logger.warning("DEPRECATION WARNING: main.py is deprecated. Use 'uvicorn app:app' for the production API server.")

app = FastAPI(title="Legal Document Search API (Deprecated)", version="1.0.0-deprecated")

try:
    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")
except Exception as e:
    logger.error(f"Failed to initialize search engine: {e}")
    engine = None

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query text.")
    top_k: int = Field(5, description="Number of results to return.")

@app.on_event("startup")
async def startup_event():
    global engine
    if engine:
        try:
            if not engine.index_path.exists() or not engine.metadata_path.exists():
                logger.warning("Index not found. Please run fetch_and_preprocess.py and search_engine.py to build it.")
            else:
                engine.load_index()
        except Exception as e:
            logger.error(f"Error loading index: {e}")

@app.post("/search")
async def search_documents(request: SearchQuery):
    if not engine or engine.index is None:
        raise HTTPException(status_code=503, detail="Search engine not ready or index not loaded.")
    
    try:
        t_start = time.perf_counter()
        results = engine.search(query=request.query, top_k=request.top_k)
        latency_ms = (time.perf_counter() - t_start) * 1000
        
        # Format response
        formatted_results = []
        for res in results:
            formatted_results.append({
                "rank": res.get("rank"),
                "chunk_id": res.get("chunk_id"),
                "court": res.get("court"),
                "text_snippet": res.get("text_chunk")[:500] + "...", 
                "extracted_citations": res.get("extracted_citations", []),
                "cosine_similarity": res.get("cosine_similarity"),
                "source_pdf": res.get("source_pdf")
            })
            
        return {
            "query": request.query,
            "latency_ms": round(latency_ms, 2),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    is_ready = engine is not None and engine.index is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "engine_ready": is_ready
    }
