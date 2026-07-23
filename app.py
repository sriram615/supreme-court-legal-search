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
# SECTION 2 ─ THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import custom search engine and validator layer
from search_engine import LegalSearchEngine
from validator import verify_contradiction_async, VerificationVerdict

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
# SECTION 4 ─ CITATION REGEX SCHEMAS & TOKENS
# ─────────────────────────────────────────────────────────────────────────────
_STATES = (
    r"(?:SC|Bom|Cal|Del|Mad|All|Ker|Guj|Pat|Raj|MP|AP|Ori|Kar|HP|PH|"
    r"J&K|Goa|Meg|Tri|Man|Nag|Lah|Sind|Hy|Mys|Utr|Jhar|Chh|Tel)"
)
_CITATION_PATTERNS: Dict[str, re.Pattern] = {
    "AIR":        re.compile(rf"\bAIR\s+\d{{4}}\s+{_STATES}\s+\d+\b", re.I),
    "SCR":        re.compile(r"\(\d{4}\)\s+\d+\s+SCR\s+\d+", re.I),
    "SCC":        re.compile(r"\(\d{4}\)\s+\d+\s+SCC(?:\s+\(Supp\))?\s+\d+", re.I),
    "SCALE":      re.compile(r"\b\d{4}\s+(?:\(\d+\)\s+)?SCALE\s+\d+\b", re.I),
    "ILR":        re.compile(
        r"\bILR\s+(?:\(\d{4}\)\s+)?\d+\s+"
        r"(?:Delhi|Bom|Cal|Mad|All|Ker|Guj|Pat|Raj|MP|AP|Ori|Kar|HP|Punjab|Haryana)\s+\d+\b",
        re.I,
    ),
    "JT":         re.compile(r"\bJT\s+\d{4}\s+\(\d+\)\s+SC\s+\d+\b", re.I),
    "SCWR":       re.compile(r"\b\d{4}\s+SCWR\s+\d+\b", re.I),
    "MLJ":        re.compile(r"\b\d{4}\s+\(\d+\)\s+MLJ\s+\d+\b", re.I),
    "CLT":        re.compile(r"\b\d{4}\s+\(\d+\)\s+CLT\s+\d+\b", re.I),
    "GLH":        re.compile(r"\b\d{4}\s+\(\d+\)\s+GLH\s+\d+\b", re.I),
    "PLR":        re.compile(r"\b\d{4}\s+(?:\(\d+\)\s+)?PLR\s+\d+\b", re.I),
    "BLJR":       re.compile(r"\b\d{4}\s+BLJR\s+\d+\b", re.I),
    "NLJ":        re.compile(r"\bNLJ\s+\d{4}\s+(?:SC\s+)?\d+\b", re.I),
    "INSC":       re.compile(r"\b\d{4}\s+INSC\s+\d+\b", re.I),
    "SLP":        re.compile(
        r"\bSLP\s+(?:\(Civil\)|\(Criminal\))?\s*No\.?\s*\d+(?:[-–]\d+)?\s+of\s+\d{4}\b",
        re.I,
    ),
    "WRIT":       re.compile(
        r"\bWrit\s+Petition\s+(?:\(Civil\)|\(Criminal\))?\s*No\.?\s*\d+\s+of\s+\d{4}\b",
        re.I,
    ),
}

def _make_token(raw: str) -> str:
    token = re.sub(r"[\s()/]+", "_", raw.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.upper()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ PYDANTIC API SCHEMAS
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

class AnalyzeResponse(BaseModel):
    contradiction_probability: float = Field(..., description="Calculated softmax probability for Class 1 (Contradiction).")
    is_contradiction: bool = Field(..., description="Binary flag indicating contradiction status (probability >= 0.5).")
    resolved_citations: Dict[str, str] = Field(..., description="Extracts and resolves canonical case citations using citation_map.json.")
    semantic_matches: List[SemanticMatch] = Field(..., description="Top 3 nearest semantic precedents from the HNSW index.")
    verification_verdict: Optional[VerificationVerdict] = Field(None, description="Referee model verification results.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ LIFESPAN STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Startup: Load Semantic Search Index ──
    logger.info("Initializing search engine state...")
    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")
    
    # Handle missing index by constructing a fast validation index from the corpus file
    if not engine.index_path.exists() or not engine.metadata_path.exists():
        corpus_path = "./LAWdata_Corpus_2024/aws_court_chunks.jsonl"
        if os.path.exists(corpus_path):
            logger.warning("Index not found. Constructing validation index from %s...", corpus_path)
            engine.build_index(corpus_path=corpus_path, hnsw_m=16, limit=200)
        else:
            raise FileNotFoundError(
                f"No HNSW index or raw corpus jsonl file found at {corpus_path}. "
                f"Please run fetch_and_preprocess.py or search_engine.py first."
            )
    else:
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

    # ── 3. Load Legal-BERT Sequence Classifier ──
    model_path = "./LAWdata_Corpus_2024/optimized_legal_bert/"
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        logger.info("Loading fine-tuned Legal-BERT classifier from local path: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        fallback_model = "nlpaueb/legal-bert-base-uncased"
        logger.warning(
            "Local classifier weights not found at %s. "
            "Downloading fallback model '%s' from Hugging Face...",
            model_path, fallback_model
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model, num_labels=2)
        
        # Cache fallback model locally for persistent offline startups
        logger.info("Caching fallback model weights to local directory: %s", model_path)
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    
    # Force model onto CPU and set evaluate mode
    model.to("cpu")
    model.eval()
    app.state.tokenizer = tokenizer
    app.state.model = model

    logger.info("All engine components warm-loaded. Ready for requests.")
    yield
    
    # ── 4. Shutdown Cleanup ──
    logger.info("Cleaning engine cache allocations...")
    del app.state.search_engine
    del app.state.model
    del app.state.tokenizer
    del app.state.citation_map

# Initialize FastAPI App with lifespan state manager
app = FastAPI(
    title="Indian Supreme Court Legal Analytics Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Legal Brief for citations, semantics, and contradictions",
)
async def analyze_legal_brief(request: AnalyzeRequest):
    """
    Run comprehensive semantic, citation, and contradiction analysis on brief snippets:
    - **Retrieval:** Fetch Top-3 matching Supreme Court precedents using FAISS HNSW.
    - **Classification:** Compute contradiction probability (Class 1) using Legal-BERT.
    - **Citations:** Parse, normalize, and resolve citations from `citation_map.json`.
    """
    try:
        t_start = time.perf_counter()
        
        # ── Stage A: Semantic Precedent Retrieval (Top-3) ──
        engine: LegalSearchEngine = app.state.search_engine
        semantic_matches = engine.search(query=request.text, top_k=3)
        
        # ── Stage B: Legal-BERT Contradiction Classification ──
        tokenizer = app.state.tokenizer
        model = app.state.model
        
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        
        # Map Class 1 as contradiction
        contradiction_prob = probs[1]
        is_contradiction = contradiction_prob >= 0.5

        # ── Stage C: Citation Resolution & Cross-Referencing ──
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

        # ── Stage D: Referee LLM Governance Validation ──
        precedent_text = ""
        if semantic_matches:
            precedent_text = semantic_matches[0].get("text_chunk", "")

        verification_verdict = None
        if precedent_text:
            verification_verdict = await verify_contradiction_async(request.text, precedent_text)

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info("API Analyze completed in %.2f ms", latency_ms)

        return AnalyzeResponse(
            contradiction_probability=contradiction_prob,
            is_contradiction=is_contradiction,
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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ─ ENTRYPOINT & EXECUTION PATTERN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local debug server execution via uvicorn...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
