"""
search_engine.py
================
Highly optimized semantic search engine for the Indian Supreme Court corpus.
Target Architecture: Intel Mac (x86_64 CPU)

Key Optimizations:
  1. NumPy & PyTorch thread constraint: torch.set_num_threads(4) to align matrix math
     natively with physical CPU cores, avoiding AVX core thrashing.
  2. Memory-Safe streaming: reads the large JSONL database in chunked batches
     of 128 rows, preventing massive memory spikes.
  3. FAISS HNSW indexing: uses IndexHNSWFlat with METRIC_INNER_PRODUCT and normalized L2
     vectors to guarantee sub-millisecond retrieval latency on Intel processors.
  4. Precomputed structured inverted indices: citation_index and section_index are built
     once at load/build time, replacing an O(N) per-query regex scan with an O(k) lookup
     for CITATION_OR_CASE_NO and STATUTORY_SECTION query intents.
  5. Fully object-oriented, typed, and production-grade error handling.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ─ STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import re
from utils import adjust_similarity_scores


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("[FATAL] numpy not installed. Run: pip install numpy") from exc

try:
    import torch
except ImportError as exc:
    raise SystemExit("[FATAL] torch not installed. Run: pip install torch") from exc

try:
    import faiss
    # Limit FAISS internal OpenMP threads to 1 to prevent deadlock on macOS CPU
    faiss.omp_set_num_threads(1)
except ImportError as exc:
    raise SystemExit("[FATAL] faiss-cpu not installed. Run: pip install faiss-cpu") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit("[FATAL] sentence-transformers not installed. Run: pip install sentence-transformers") from exc

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise SystemExit("[FATAL] rank-bm25 not installed. Run: pip install rank-bm25") from exc

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", 
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

PROCEDURAL_VOCAB = {
    "court", "high", "order", "state", "case", "appellant", "appellants",
    "respondent", "respondents", "matter", "learned", "counsel", "jurisdiction",
    "judgment", "judgement", "held", "paras", "honble", "bench", "division",
    "leave", "granted", "heard", "parties", "petition", "writ", "appeal",
    "claims", "litigation", "procedure", "offences", "offence"
}

FOREIGN_JURISDICTION_VOCAB = {
    "uncitral", "singapore", "federal", "district", "london", "uk", "usa",
    "us", "eu", "european", "american", "foreign"
}

def _tokenize_legal_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extract alphanumeric tokens and citation components for BM25 lexical indexing."""
    raw_tokens = re.findall(r"\b[A-Za-z0-9_]+\b", text.lower())
    if remove_stopwords:
        return [t for t in raw_tokens if t not in ENGLISH_STOP_WORDS]
    return raw_tokens

def classify_and_extract_query(query: str) -> Dict[str, Any]:
    """
    Categorize incoming user query into one of 4 intent categories and extract structural entities.
    
    Query Categories:
      1. 'CITATION_OR_CASE_NO': Queries containing neutral citations, reporter citations, or case numbers.
      2. 'CASE_TITLE': Queries containing party names ('v.', 'vs.', 'State of').
      3. 'STATUTORY_SECTION': Queries containing statutory sections ('Section 14(1)(j-1)', 'Section 271(1)(c)').
      4. 'CONCEPTUAL_OR_AMBIGUOUS': Conceptual legal queries, paraphrases, ambiguous terms, or OOD inputs.
    """
    q_lower = query.lower().strip()
    
    # ── 1. Check for Citations or Case Numbers ──
    case_no_match = re.search(r"\b(?:slp|civil appeal|writ petition|appeal|petition|no\.?)\b.*?\b(\d{3,6})\b", q_lower)
    cit_insc_match = re.search(r"\b(\d{4}_insc_\d+|\d{4}\s*insc\s*\d+)\b", q_lower)
    cit_historic_match = re.search(r"\((19\d{2}|20\d{2})\)\s*(\d*)\s*(scr|scc|air)\s*(\d+)", q_lower)
    cit_reporter_match = re.search(r"\b(air|\d*\s*scr|\d*\s*scc)\b", q_lower)

    extracted_identifiers = []
    if cit_insc_match:
        raw_cit = cit_insc_match.group(1).replace(" ", "_")
        extracted_identifiers.append(raw_cit)
    elif cit_historic_match:
        yr, vol, rep, pg = cit_historic_match.groups()
        norm_cit = f"{yr}_{vol}_{rep}_{pg}" if vol else f"{yr}_{rep}_{pg}"
        space_cit = f"{vol} {rep} {pg}".strip() if vol else f"{rep} {pg}".strip()
        extracted_identifiers.extend([norm_cit, space_cit])
    elif case_no_match:
        num = case_no_match.group(1)
        if not (len(num) == 4 and 1900 <= int(num) <= 2099):
            extracted_identifiers.append(num)
    elif cit_reporter_match:
        nums = re.findall(r"\b\d{1,5}\b", q_lower)
        valid_nums = [n for n in nums if not (len(n) == 4 and 1900 <= int(n) <= 2099) and int(n) >= 10]
        if valid_nums:
            extracted_identifiers.extend(valid_nums)

    if extracted_identifiers or cit_historic_match or "insc" in q_lower or ("no." in q_lower and re.search(r"\d{3,}", q_lower)):
        return {
            "query_type": "CITATION_OR_CASE_NO",
            "extracted_identifiers": extracted_identifiers,
            "raw_query": query,
        }

    # ── 2. Check for Case Titles (Party Names) ──
    if " v. " in q_lower or " vs. " in q_lower or " vs " in q_lower or " versus " in q_lower or " v " in q_lower:
        generic_title_words = {"v", "vs", "versus", "state", "of", "the", "and", "ors", "anr", "govt", "government", "union", "india", "pvt", "ltd", "co", "corp"}
        jurisdiction_state_words = {
            "state", "union", "india", "kerala", "maharashtra", "haryana", "bihar", "delhi",
            "punjab", "gujarat", "karnataka", "tamil", "nadu", "up", "mp", "ap", "rajasthan",
            "bengal", "orissa", "odisha", "assam", "telangana", "andhra", "pradesh"
        }
        raw_words = re.findall(r"\b[A-Za-z0-9_]+\b", q_lower)
        party_nouns = [w for w in raw_words if w not in ENGLISH_STOP_WORDS and w not in generic_title_words and len(w) >= 3]
        primary_party_nouns = [w for w in party_nouns if w not in jurisdiction_state_words and len(w) >= 3]
        return {
            "query_type": "CASE_TITLE",
            "extracted_party_nouns": party_nouns,
            "primary_party_nouns": primary_party_nouns if primary_party_nouns else party_nouns,
            "raw_query": query,
        }

    # ── 3. Check for Statutory Section Queries ──
    sec_match = re.search(r"\b(?:section|sec\.?|article|art\.?|s\.?)\s*([0-9a-zA-Z\(\)\-\.]+)", q_lower)
    if sec_match and re.search(r"\d", sec_match.group(1)):
        full_sec = sec_match.group(1)
        digits_match = re.search(r"^\d+", full_sec)
        sec_digit = digits_match.group(0) if digits_match else full_sec
        return {
            "query_type": "STATUTORY_SECTION",
            "full_section_str": full_sec,
            "section_digit": sec_digit,
            "raw_query": query,
        }

    # ── 4. Default: Conceptual, Ambiguous, or OOD Queries ──
    content_tokens = _tokenize_legal_text(query, remove_stopwords=True)
    return {
        "query_type": "CONCEPTUAL_OR_AMBIGUOUS",
        "content_tokens": content_tokens,
        "raw_query": query,
    }

def _is_statutory_section_present(full_sec: str, sec_digit: str, doc_text: str) -> bool:
    """
    Verify if full_sec (e.g. '304b', '14(1)(j-1)', '438', '302') is contiguously present in doc_text.
    Does NOT allow a bare numeric prefix (e.g. '304') to satisfy a query requiring a sub-letter (e.g. '304b').
    """
    doc_lower = doc_text.lower()
    full_sec_lower = full_sec.lower().strip()

    # 1. Exact string match of full_sec in document text (e.g., '304b', '304-b', '14(1)(j-1)', '438')
    if full_sec_lower in doc_lower:
        return True

    # 2. Check for alpha sub-section suffix (e.g. '304b' -> '304-b', '304 (b)', '304(b)')
    m = re.match(r"^(\d+)[\-\s]*([a-zA-Z]+)$", full_sec_lower)
    if m:
        digits, letters = m.group(1), m.group(2)
        pattern = r"\b" + digits + r"[\s\-\(\)]*" + letters + r"\b"
        return bool(re.search(pattern, doc_lower))

    # 3. For numeric sections (or complex clause sections like 14(1)(j-1) where digit is 14),
    # verify sec_digit is present as a standalone numeric token in doc_text
    if sec_digit:
        pattern = r"\b" + sec_digit + r"\b"
        return bool(re.search(pattern, doc_lower))

    return False

def evaluate_routed_relevance(
    hit: Dict[str, Any],
    query: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Routed Multi-Stage Metadata Verification Relevance Gate.
    Separates query intent into structural categories and evaluates deterministic metadata verification rules.
    """
    if config is None:
        config = {
            "conceptual_semantic_floor": 0.45,
            "conceptual_min_matched_tokens": 2,
            "conceptual_bm25_floor": 4.0,
        }

    if not hit:
        return {
            "is_relevant": False,
            "query_type": "UNKNOWN",
            "meta_verified": False,
            "matched_meaningful_tokens": [],
            "gate_reason": "No candidate hit returned from retrieval pipeline",
        }

    parsed = classify_and_extract_query(query)
    q_type = parsed["query_type"]
    
    cosine_sim = hit.get("cosine_similarity", 0.0)
    bm25_score = hit.get("bm25_score", 0.0)
    
    text_chunk = hit.get("text_chunk", "")
    pdf_name = hit.get("source_pdf", "")
    chunk_id = hit.get("chunk_id", "")
    citations_list = hit.get("extracted_citations", [])
    citations_str = " ".join(citations_list).lower()
    header_text = text_chunk[:400].lower()
    full_doc_text = text_chunk.lower()
    
    doc_tokens = set(_tokenize_legal_text(f"{text_chunk} {pdf_name} {chunk_id} {citations_str}", remove_stopwords=True))
    
    # ── ROUTE A: CITATION OR CASE NUMBER QUERY ──
    if q_type == "CITATION_OR_CASE_NO":
        target_ids = parsed.get("extracted_identifiers", [])
        verified = False
        matched_tokens = []
        
        for tid in target_ids:
            tid_lower = tid.lower()
            pattern = r"\b" + re.escape(tid_lower) + r"\b"
            if (
                re.search(pattern, citations_str)
                or re.search(pattern, header_text)
                or re.search(pattern, pdf_name.lower())
                or re.search(pattern, full_doc_text)
            ):
                verified = True
                matched_tokens.append(tid)
                
        if verified:
            return {
                "is_relevant": True,
                "query_type": q_type,
                "meta_verified": True,
                "matched_meaningful_tokens": matched_tokens,
                "gate_reason": f"Exact Citation/Case Number verified in metadata/header: {matched_tokens}",
            }
        else:
            return {
                "is_relevant": False,
                "query_type": q_type,
                "meta_verified": False,
                "matched_meaningful_tokens": [],
                "gate_reason": f"Requested identifier(s) {target_ids} absent from document metadata/header",
            }

    # ── ROUTE B: CASE TITLE QUERY ──
    elif q_type == "CASE_TITLE":
        party_nouns = parsed.get("extracted_party_nouns", [])
        primary_nouns = parsed.get("primary_party_nouns", party_nouns)

        matched_tokens = [p for p in party_nouns if p in header_text or p in full_doc_text]
        primary_matched = [p for p in primary_nouns if p in header_text or p in full_doc_text]

        # Case title verification requires matching at least one primary party entity
        if primary_matched:
            return {
                "is_relevant": True,
                "query_type": q_type,
                "meta_verified": True,
                "matched_meaningful_tokens": matched_tokens,
                "gate_reason": f"Case Title primary party entities verified in document: {primary_matched}",
            }
        else:
            return {
                "is_relevant": False,
                "query_type": q_type,
                "meta_verified": False,
                "matched_meaningful_tokens": [],
                "gate_reason": f"Primary party entities {primary_nouns} absent from candidate document header/text",
            }

    # ── ROUTE C: STATUTORY SECTION QUERY ──
    elif q_type == "STATUTORY_SECTION":
        sec_digit = parsed.get("section_digit", "")
        full_sec = parsed.get("full_section_str", "")

        sec_present = _is_statutory_section_present(full_sec, sec_digit, full_doc_text)

        if sec_present:
            return {
                "is_relevant": True,
                "query_type": q_type,
                "meta_verified": True,
                "matched_meaningful_tokens": [full_sec],
                "gate_reason": f"Specific statutory section '{full_sec}' verified in document text",
            }
        else:
            return {
                "is_relevant": False,
                "query_type": q_type,
                "meta_verified": False,
                "matched_meaningful_tokens": [],
                "gate_reason": f"Specific statutory section '{full_sec}' absent from document text",
            }

    # ── ROUTE D: CONCEPTUAL, AMBIGUOUS, OR OOD QUERY ──
    else:
        content_tokens = parsed.get("content_tokens", [])
        q_lower_words = set(re.findall(r"\b[A-Za-z0-9_]+\b", query.lower()))

        # Check for foreign jurisdiction disqualification
        foreign_matched = [w for w in q_lower_words if w in FOREIGN_JURISDICTION_VOCAB and w not in doc_tokens]
        if foreign_matched and cosine_sim < 0.45:
            return {
                "is_relevant": False,
                "query_type": q_type,
                "meta_verified": False,
                "matched_meaningful_tokens": [],
                "gate_reason": f"Foreign jurisdiction tokens {foreign_matched} present with low FAISS cosine {cosine_sim:.4f} < 0.45",
            }

        # Substantive query tokens (strip English stops + procedural vocab)
        substantive_q_tokens = [t for t in content_tokens if t not in PROCEDURAL_VOCAB]
        matched_substantive = [t for t in substantive_q_tokens if t in doc_tokens]
        n_sub = len(matched_substantive)

        # Path 1: Strong Semantic Match (FAISS >= 0.40 AND >= 1 substantive token)
        sem_pass = (cosine_sim >= 0.40) and (n_sub >= 1)

        # Path 2: Lexical / BM25 Match (with semantic floor >= 0.25 to reject pure non-legal OOD)
        lex_pass_2a = (cosine_sim >= 0.30) and (bm25_score >= 4.0) and (n_sub >= 2)
        lex_pass_2b = (cosine_sim >= 0.25) and (bm25_score >= 6.0) and (n_sub >= 1)
        lex_pass = lex_pass_2a or lex_pass_2b

        is_relevant = sem_pass or lex_pass

        reasons = []
        if sem_pass:
            reasons.append(f"Semantic pass (Cosine {cosine_sim:.4f} >= 0.40, substantive matched: {n_sub} >= 1)")
        if lex_pass:
            reasons.append(f"Lexical pass (BM25 {bm25_score:.4f}, substantive matched: {n_sub})")

        gate_reason = " | ".join(reasons) if is_relevant else (
            f"Failed conceptual gate (Sim {cosine_sim:.4f}, BM25 {bm25_score:.4f}, substantive matched: {n_sub})"
        )

        return {
            "is_relevant": is_relevant,
            "query_type": q_type,
            "meta_verified": is_relevant,
            "matched_meaningful_tokens": matched_substantive,
            "gate_reason": gate_reason,
        }

def evaluate_relevance(
    hit: Dict[str, Any],
    query: str,
    semantic_threshold: float = 0.35,
    lexical_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Compatibility wrapper calling evaluate_routed_relevance."""
    return evaluate_routed_relevance(hit, query)




# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("search_engine.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SC_Search")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ THREAD OPTIMIZATION GATES
# ─────────────────────────────────────────────────────────────────────────────
# Force torch to run on a maximum of 4 physical cores to maximize AVX2 efficiency.
# This prevents hyperthreading overhead and core contention on Intel Core i7/i9 processors.
torch.set_num_threads(4)
logger.info("Intel Mac optimization: set torch thread count to 4.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ CENTRAL SEARCH ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LegalSearchEngine:
    """
    Central search infrastructure managing Sentence-Transformer embeddings,
    memory-efficient batch processing, and accelerated FAISS HNSW graph indexing.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = "./LAWdata_Corpus_2024",
    ) -> None:
        """
        Initialize the search engine with explicit model name and target folders.

        Args:
            model_name: Name of the SentenceTransformer model to load (dimension 384).
            index_dir: The directory where the FAISS index and metadata mappings reside.
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "legal_precedents.index"
        self.metadata_path = self.index_dir / "metadata_map.json"

        # Explicitly enforce CPU device mapping
        self.device = "cpu"
        logger.info("Loading transformer model: %s on device: %s", self.model_name, self.device)
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Ensure model runs with CPU thread tuning
            self.model.to(self.device)
            logger.info("Model loaded successfully. Embedding dimension: %d", self.model.get_sentence_embedding_dimension())
        except Exception as exc:
            logger.error("Failed to load transformer model %s: %s", self.model_name, exc)
            raise

        # Internal index and metadata references
        self.index: Optional[faiss.IndexHNSWFlat] = None
        self.metadata_map: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        # Precomputed structured inverted indices (built once in _build_structured_indices)
        self.citation_index: Dict[str, List[int]] = {}   # token → [metadata_map indices]
        self.section_index: Dict[str, List[int]] = {}    # sec_digit → [metadata_map indices]
        self.title_index: Dict[str, List[int]] = {}      # party token → [metadata_map indices]

    def _init_bm25_index(self) -> None:
        """Initialize BM25 lexical search index over metadata text chunks and citations."""
        t0 = time.perf_counter()
        corpus_tokens = []
        for meta in self.metadata_map:
            chunk_text = meta.get("text_chunk", "")
            pdf_name = meta.get("source_pdf", "")
            chunk_id = meta.get("chunk_id", "")
            citations = " ".join(meta.get("extracted_citations", []))
            full_text = f"{chunk_text} {pdf_name} {chunk_id} {citations}"
            tokens = _tokenize_legal_text(full_text)
            corpus_tokens.append(tokens)
        self.bm25 = BM25Okapi(corpus_tokens)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("BM25 lexical index ready (%d chunks) in %.2f ms.", len(corpus_tokens), elapsed)

    def _build_structured_indices(self) -> None:
        """
        Precompute two inverted lookup indices over self.metadata_map:

        citation_index  : Dict[token_str, List[int]]
            Maps lowercase alphanumeric tokens (excluding English function words) from
            each chunk's extracted_citations, source_pdf, and first-400-char header.
            Additionally indexes 3-6-digit numeric tokens from the FULL text body so that
            case numbers appearing beyond the header (e.g. "12326" in SLP queries) are
            still found by the O(k) lookup.

        section_index   : Dict[sec_digit, List[int]]
            Maps statutory section digit strings (e.g. '304', '138', '482', '14') to
            metadata indices using STANDALONE numeric tokens only.

            Tokens are matched with r'\b(\d+[a-zA-Z]?)\b' so that:
              - "304b", "304B" → indexed under "304"
              - "482"          → indexed under "482"
              - "14" in "2014" → NOT indexed under "14"  (word boundary prevents it)
              - "14" standalone → indexed under "14" ✓

            This makes short section numbers like "14", "19", "21" selective: they are
            only indexed for chunks where the number appears as a genuine standalone token,
            not embedded inside year numbers or case numbers.

            Rule #2 / posting-list selectivity:
              - 3-digit sections (304, 482, 138): typically 1-3% of corpus ✓
              - 2-digit sections (14, 19, 21): 15-22% — still >5% because these
                genuinely appear standalone (para numbers, article refs, etc.).
                This is an inherent limit; the runtime filter (condition B only
                for numeric-only sections) narrows results further.

        title_index     : Dict[party_token, List[int]]
            Maps party name tokens from the header and source_pdf to metadata indices.

        Both dicts are built in a single O(N) pass and are never rebuilt unless
        load_index() or build_index() is called again.
        """
        t0 = time.perf_counter()
        citation_index: Dict[str, List[int]] = {}
        section_index: Dict[str, List[int]] = {}
        title_index: Dict[str, List[int]] = {}

        # English function words AND domain-generic legal terms that can never
        # be citation identifier tokens.
        # Filtering these eliminates posting lists covering 14–100% of corpus.
        # Domain words added after the §8b audit showed 'under' (22%), 'court' (39%),
        # 'act' (19%), 'section' (18%), 'state' (18%), 'case' (18%), 'order' (14%)
        # all surviving the original function-word-only stop list.
        _CITATION_STOP: frozenset = frozenset({
            # English function words
            "the", "of", "to", "and", "in", "is", "for", "it", "be", "as",
            "by", "we", "at", "or", "an", "this", "that", "was", "are", "not",
            "but", "had", "on", "its", "if", "from", "with", "has", "he", "she",
            "his", "her", "they", "our", "can", "no", "do", "so", "up", "out",
            "my", "all", "may", "per", "into", "then", "than", "been", "who",
            "will", "also", "any", "which", "when", "where", "how", "would",
            "after", "about", "being", "their", "have", "were", "other",
            # Domain-generic legal words (high corpus coverage, never citation sub-tokens)
            "under", "court", "act", "section", "state", "case", "order",
            "held", "filed", "bench", "petition", "judgment", "appeal",
        })

        _TITLE_STOP: frozenset = _CITATION_STOP.union({
            "v", "vs", "versus", "union", "india", "pdf", "others", "another"
        })

        # Standalone numeric token pattern — does NOT match digits inside larger numbers
        _STANDALONE_NUM = re.compile(r"\b(\d+[a-zA-Z]?)\b")

        for idx, meta in enumerate(self.metadata_map):
            txt_lower = meta.get("text_chunk", "").lower()
            pdf_lower = meta.get("source_pdf", "").lower()
            header    = txt_lower[:400]
            cits_str  = " ".join(meta.get("extracted_citations", [])).lower()

            # ── Citation index ──
            # Source: extracted_citations, first-400-char header, source_pdf basename.
            # Stop-word filter removes function words that would create ~100% posting lists.
            combined_cit = f"{cits_str} {header} {pdf_lower}"
            for tok in re.findall(r"[a-z0-9_]+", combined_cit):
                if len(tok) >= 2 and tok not in _CITATION_STOP:
                    if tok not in citation_index:
                        citation_index[tok] = []
                    citation_index[tok].append(idx)

            # Full-text body: 3-6-digit numeric tokens (case numbers buried in body text).
            for tok in re.findall(r"\d+", txt_lower):
                if 3 <= len(tok) <= 6:
                    if tok not in citation_index:
                        citation_index[tok] = []
                    citation_index[tok].append(idx)

            # ── Section index (standalone tokens only) ──
            # \b(\d+[a-zA-Z]?)\b matches "304b", "482", "14" as standalone words.
            # "14" inside "2014" is NOT matched (no word boundary between digits).
            for m in _STANDALONE_NUM.finditer(txt_lower):
                num_tok = m.group(1)
                dm = re.match(r"^\d+", num_tok)
                if dm:
                    key = dm.group(0)
                    if key not in section_index:
                        section_index[key] = []
                    section_index[key].append(idx)

            # ── Title index (party tokens from header & source_pdf) ──
            header_and_pdf = f"{header} {pdf_lower}"
            for tok in re.findall(r"\b[a-z]{3,}\b", header_and_pdf):
                if tok not in _TITLE_STOP:
                    if tok not in title_index:
                        title_index[tok] = []
                    title_index[tok].append(idx)

        # Deduplicate while preserving insertion order
        self.citation_index = {k: list(dict.fromkeys(v)) for k, v in citation_index.items()}
        self.section_index  = {k: list(dict.fromkeys(v)) for k, v in section_index.items()}
        self.title_index    = {k: list(dict.fromkeys(v)) for k, v in title_index.items()}

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Structured inverted indices built: %d citation tokens, %d section digits in %.2f ms.",
            len(self.citation_index), len(self.section_index), elapsed,
        )

    def build_index(
        self,
        corpus_path: str = "./LAWdata_Corpus_2024/aws_court_chunks.jsonl",
        batch_size: int = 128,
        hnsw_m: int = 16,
        limit: Optional[int] = None,
    ) -> None:
        """
        Stream rows from the JSONL database, encode them into dense vectors in
        memory-safe batches, and build an accelerated FAISS HNSW index.

        Args:
            corpus_path: Path to the input JSONL preprocessed chunks file.
            batch_size: Batch sizing to prevent RAM spikes (default 128).
            hnsw_m: Number of connections per node in HNSW graph (default 16).
            limit: If set, restricts indexing to the first N chunks (e.g. for quick validation).
        """
        t_start = time.perf_counter()
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file.resolve()}")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        npy_cache_path = self.index_dir / "embeddings.npy"

        # Check if pre-computed embeddings and metadata map exist
        if limit is None and npy_cache_path.exists() and self.metadata_path.exists():
            logger.info("Found cached embeddings at %s. Fast loading cached state...", npy_cache_path)
            embeddings_matrix = np.load(str(npy_cache_path)).astype("float32")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata_map = json.load(f)
            total_rows_processed = len(self.metadata_map)
        else:
            logger.info("Building HNSW semantic index from: %s", corpus_file.resolve())
            if limit is not None:
                logger.info("Dry-run build: limit set to first %d rows.", limit)

            # Collect text fragments, parallel chunk IDs, and full metadata mappings
            text_batch: List[str] = []
            meta_batch: List[Dict[str, Any]] = []
            all_embeddings: List[np.ndarray] = []

            total_rows_processed = 0

            # Read JSONL database iteratively in chunks
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        text_chunk = row.get("text_chunk", "").strip()
                        if not text_chunk:
                            continue

                        # Maintain a clean copy of metadata mapping
                        metadata = {
                            "chunk_id": row.get("chunk_id", ""),
                            "parent_judgment_id": row.get("parent_judgment_id", ""),
                            "court": row.get("court", "Supreme Court of India"),
                            "tokens_count": row.get("tokens_count", 0),
                            "text_chunk": text_chunk,
                            "extracted_citations": row.get("extracted_citations", []),
                            "language_flag": row.get("language_flag", "en"),
                            "source_pdf": row.get("source_pdf", ""),
                        }

                        text_batch.append(text_chunk)
                        meta_batch.append(metadata)

                        # Trigger batch encoding when size matches limit
                        if len(text_batch) == batch_size:
                            embeddings = self.model.encode(
                                text_batch,
                                batch_size=batch_size,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True,  # Normalized L2 vectors for cosine similarity
                            )
                            all_embeddings.append(embeddings)
                            self.metadata_map.extend(meta_batch)
                            total_rows_processed += len(text_batch)
                            logger.info("  Processed %d rows...", total_rows_processed)

                            # Clean batch references
                            text_batch.clear()
                            meta_batch.clear()

                        # Exit if limit reached
                        if limit is not None and total_rows_processed >= limit:
                            break

                    except json.JSONDecodeError as exc:
                        logger.warning("Malformed JSON row ignored: %s", exc)
                    except Exception as exc:
                        logger.error("Error processing line: %s", exc)

                # Process final remaining lines
                if text_batch and (limit is None or total_rows_processed < limit):
                    # Cut batch size if it exceeds remaining limit
                    if limit is not None:
                        remaining = limit - total_rows_processed
                        text_batch = text_batch[:remaining]
                        meta_batch = meta_batch[:remaining]

                    if text_batch:
                        embeddings = self.model.encode(
                            text_batch,
                            batch_size=len(text_batch),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                        all_embeddings.append(embeddings)
                        self.metadata_map.extend(meta_batch)
                        total_rows_processed += len(text_batch)

            if not all_embeddings:
                raise ValueError("No valid text embeddings were generated from the corpus database.")

            # Stack batch embeddings into a single dense matrix
            embeddings_matrix = np.vstack(all_embeddings).astype("float32")

            # Cache computed embeddings to disk for future fast loads (if not a limited build)
            if limit is None:
                logger.info("Caching dense embeddings matrix to disk: %s", npy_cache_path)
                np.save(str(npy_cache_path), embeddings_matrix)

        dimension = embeddings_matrix.shape[1]
        logger.info("Generated dense matrix shape: %s", embeddings_matrix.shape)
        logger.info("Initializing FAISS HNSW Flat Index (dimension: %d, Metric: METRIC_INNER_PRODUCT)", dimension)

        # Initialize HNSW index utilizing Inner Product (which behaves as Cosine Similarity over L2 normalized vectors)
        # M = hnsw_m (connections per node), construction parameter is default
        self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        
        # Configure search graph parameters
        self.index.hnsw.efSearch = 64
        self.index.hnsw.efConstruction = 128

        logger.info("Adding embedding matrix to FAISS index...")
        self.index.add(embeddings_matrix)

        # Write to disk
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_map, f, indent=2, ensure_ascii=False)

        # Initialize BM25 lexical index
        self._init_bm25_index()

        # Precompute structured inverted indices for O(1) citation/section lookups
        self._build_structured_indices()

        elapsed = time.perf_counter() - t_start
        logger.info("=" * 70)
        logger.info("  INDEX BUILDING COMPLETE")
        logger.info("  Total rows indexed   : %d", total_rows_processed)
        logger.info("  Time elapsed         : %.1f seconds", elapsed)
        logger.info("  Serialized index to  : %s", self.index_path)
        logger.info("  Serialized metadata  : %s", self.metadata_path)
        logger.info("=" * 70)

    def load_index(self) -> None:
        """
        Load the serialized FAISS HNSW graph index and matching metadata map from disk.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata mapping file not found at: {self.metadata_path}")

        logger.info("Loading FAISS index from: %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

        logger.info("Loading metadata maps from: %s", self.metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata_map = json.load(f)

        logger.info("Index loaded. Total vectors inside: %d", self.index.ntotal)
        self._init_bm25_index()

        # Precompute structured inverted indices for O(1) citation/section lookups
        self._build_structured_indices()

    def _get_exact_identifier_indices(self, parsed: Dict[str, Any]) -> Set[int]:
        """Extract exact identifier candidate indices for structured query intents."""
        qtype = parsed.get("query_type", "CONCEPTUAL_OR_AMBIGUOUS")
        exact_identifier_indices: Set[int] = set()
        if qtype == "CITATION_OR_CASE_NO":
            target_ids = parsed.get("extracted_identifiers", [])
            for tid in target_ids:
                sub_tokens = re.findall(r"[a-z0-9_]+", tid.lower())
                for tok in sub_tokens:
                    if len(tok) >= 2 and tok in self.citation_index:
                        pat = r"\b" + re.escape(tid.lower()) + r"\b"
                        for candidate_idx in self.citation_index[tok]:
                            meta = self.metadata_map[candidate_idx]
                            txt    = meta.get("text_chunk", "").lower()
                            pdf    = meta.get("source_pdf", "").lower()
                            cits   = " ".join(meta.get("extracted_citations", [])).lower()
                            header = txt[:400]
                            if (
                                re.search(pat, cits)
                                or re.search(pat, header)
                                or re.search(pat, pdf)
                                or re.search(pat, txt)
                            ):
                                exact_identifier_indices.add(candidate_idx)

        elif qtype == "CASE_TITLE":
            primary_nouns = parsed.get("primary_party_nouns", [])
            if primary_nouns:
                for p in primary_nouns:
                    if p in self.title_index:
                        for candidate_idx in self.title_index[p]:
                            exact_identifier_indices.add(candidate_idx)

        elif qtype == "STATUTORY_SECTION":
            sec_digit = parsed.get("section_digit", "")
            full_sec  = parsed.get("full_section_str", "").lower()
            raw_q     = parsed.get("raw_query", "").lower()
            if sec_digit and sec_digit in self.section_index:
                _full_sec_has_letter = bool(full_sec and re.search(r"[a-zA-Z]", full_sec))
                if _full_sec_has_letter:
                    for candidate_idx in self.section_index[sec_digit]:
                        txt = self.metadata_map[candidate_idx].get("text_chunk", "").lower()
                        if _is_statutory_section_present(full_sec, sec_digit, txt):
                            exact_identifier_indices.add(candidate_idx)
                else:
                    is_article = bool(re.search(r"\b(?:article|art\.?)\b", raw_q))
                    prefix_pat = r"\b(?:article|art\.?)\s*" if is_article else r"\b(?:section|sec\.?|s\.?|u/s\.?|u/ss\.?)\s*"
                    _sec_pat_fast = re.compile(prefix_pat + re.escape(sec_digit) + r"\b", re.I)
                    for candidate_idx in self.section_index[sec_digit]:
                        txt = self.metadata_map[candidate_idx].get("text_chunk", "").lower()
                        if _sec_pat_fast.search(txt):
                            exact_identifier_indices.add(candidate_idx)

        return exact_identifier_indices

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_year: int = 2026,
        decay_lambda: float = 0.005,
        mode: str = "hybrid",
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Search legal database using hybrid (BM25 + FAISS RRF), FAISS dense vector search,
        or BM25 lexical search.

        Args:
            query: The search query.
            top_k: Number of results to return.
            query_year: Query year for age decay calculation.
            decay_lambda: Age decay penalty lambda.
            mode: Search mode: 'hybrid' (default RRF), 'faiss' (dense vector), or 'bm25' (lexical).
            rrf_k: Reciprocal Rank Fusion constant (default 60).

        Returns:
            List[Dict[str, Any]]: Search hits with rankings, scores, and metadata.
        """
        t_start = time.perf_counter()
        if self.index is None or not self.metadata_map:
            logger.info("Cold starting / loading index from default location...")
            self.load_index()

        assert self.index is not None, "FAISS index failed to initialize."
        if self.bm25 is None:
            self._init_bm25_index()

        # ── 1. Parse Query Intent & Structure ──
        parsed = classify_and_extract_query(query)
        qtype = parsed.get("query_type", "CONCEPTUAL_OR_AMBIGUOUS")

        # ── 2. Dense Semantic Vector Search (FAISS) ──
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        candidate_k = max(top_k * 10, 50)
        D, I = self.index.search(query_vector, candidate_k)

        faiss_ranks: Dict[int, int] = {}
        faiss_scores: Dict[int, float] = {}
        for rank_idx, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            if idx >= 0 and idx < len(self.metadata_map):
                faiss_ranks[idx] = rank_idx
                faiss_scores[idx] = float(np.clip(score, -1.0, 1.0))

        # ── 3. Lexical Keyword Search (BM25) ──
        query_tokens = _tokenize_legal_text(query)
        bm25_scores_all = self.bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(self.metadata_map))
        bm25_idx_sorted = list(np.argsort(bm25_scores_all)[::-1])

        # ── 4. Structured Query Intent Candidate Pool & Exact Identifier Boost ──
        # Uses precomputed inverted indices (O(k) lookup) instead of the previous
        # O(N) per-query scan over all metadata entries.
        exact_identifier_indices = self._get_exact_identifier_indices(parsed)

        # Prioritize exact identifier candidates at top of BM25 lexical candidate ordering
        if exact_identifier_indices:
            exact_list = sorted(list(exact_identifier_indices), key=lambda idx: bm25_scores_all[idx], reverse=True)
            non_exact = [idx for idx in bm25_idx_sorted if idx not in exact_identifier_indices]
            ordered_bm25 = exact_list + non_exact
        else:
            ordered_bm25 = bm25_idx_sorted

        bm25_ranks: Dict[int, int] = {}
        bm25_scores: Dict[int, float] = {}
        for rank_idx, idx in enumerate(ordered_bm25[:candidate_k], start=1):
            s = float(bm25_scores_all[idx])
            if s > 0 or idx in exact_identifier_indices:
                bm25_ranks[idx] = rank_idx
                bm25_scores[idx] = s

        # ── 5. Ranking Mode Execution (RRF Fusion) ──
        results: List[Dict[str, Any]] = []

        if mode == "faiss":
            for idx, rank_val in faiss_ranks.items():
                meta = self.metadata_map[idx].copy()
                meta["cosine_similarity"] = faiss_scores[idx]
                meta["faiss_rank"] = rank_val
                meta["bm25_rank"] = bm25_ranks.get(idx, None)
                meta["bm25_score"] = bm25_scores.get(idx, 0.0)
                results.append(meta)
            results = adjust_similarity_scores(results, query_year=query_year, decay_lambda=decay_lambda)[:top_k]

        elif mode == "bm25":
            for idx, rank_val in bm25_ranks.items():
                meta = self.metadata_map[idx].copy()
                meta["cosine_similarity"] = faiss_scores.get(idx, 0.0)
                meta["faiss_rank"] = faiss_ranks.get(idx, None)
                meta["bm25_rank"] = rank_val
                meta["bm25_score"] = bm25_scores[idx]
                results.append(meta)
            results.sort(key=lambda x: x["bm25_score"], reverse=True)
            results = results[:top_k]
            for r_idx, item in enumerate(results, start=1):
                item["rank"] = r_idx

        else:  # "hybrid" (RRF)
            all_candidate_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))

            for idx in all_candidate_indices:
                meta = self.metadata_map[idx].copy()
                f_rank = faiss_ranks.get(idx, None)
                b_rank = bm25_ranks.get(idx, None)

                # Intent-aware Reciprocal Rank Fusion (RRF) Formula
                rrf_score = 0.0
                if f_rank is not None:
                    rrf_score += 1.0 / (rrf_k + f_rank)
                if b_rank is not None:
                    # Apply exact-identifier candidate weight (2.5x) for structured query intent matches
                    weight = 2.5 if (qtype in ("CITATION_OR_CASE_NO", "CASE_TITLE", "STATUTORY_SECTION") and idx in exact_identifier_indices) else 1.0
                    rrf_score += weight / (rrf_k + b_rank)

                meta["rrf_score"] = float(rrf_score)
                meta["faiss_rank"] = f_rank
                meta["bm25_rank"] = b_rank
                meta["bm25_score"] = bm25_scores.get(idx, 0.0)
                meta["cosine_similarity"] = faiss_scores.get(idx, 0.0)
                results.append(meta)

            results.sort(key=lambda x: x["rrf_score"], reverse=True)
            results = results[:top_k]
            for r_idx, item in enumerate(results, start=1):
                item["rank"] = r_idx

        latency_ms = (time.perf_counter() - t_start) * 1000
        top_score = results[0]["rrf_score"] if (results and "rrf_score" in results[0]) else (results[0]["cosine_similarity"] if results else 0.0)
        logger.info("Query [%s] retrieved in %.2f ms | Top score: %.4f", mode, latency_ms, top_score)
        return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ ENTRYPOINT & INTEGRATION VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Initializing search engine test pipeline...")
    
    # 1. Initialize engine
    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")

    # 2. Check if index files exist; if not, build it.
    if not engine.index_path.exists() or not engine.metadata_path.exists():
        logger.info("Serialized index not found on disk. Building from scratch...")
        engine.build_index()
    else:
        logger.info("Index detected on disk. Loading cached state.")
        engine.load_index()

    # 3. Perform a demonstration search
    demo_query = "What is the disqualification criteria for a female Sarpanch who encroached on government land?"
    logger.info("Executing demonstration semantic search query: '%s'", demo_query)
    
    hits = engine.search(demo_query, top_k=3)
    
    print("\n" + "=" * 80)
    print(f"SEARH HITS FOR: '{demo_query}'")
    print("=" * 80)
    for hit in hits:
        print(f"Rank {hit['rank']} (Score: {hit['cosine_similarity']:.4f})")
        print(f"  Chunk ID   : {hit['chunk_id']}")
        print(f"  Source PDF : {hit['source_pdf']}")
        print(f"  Snippet    : {hit['text_chunk'][:150]}...")
        print("-" * 80)
    print("=" * 80 + "\n")
