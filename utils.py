"""
utils.py
========
Centralized legal metadata, citation regex schemas, token formatting,
and chronological decay functions shared across the pipeline.
"""

import re
from typing import Dict, Any, List

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

def make_token(raw: str) -> str:
    """Normalize raw citation string to uniform token string (e.g. 'AIR 2024 SC 123' -> 'AIR_2024_SC_123')."""
    token = re.sub(r"[\s()/]+", "_", raw.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token.upper()

def extract_year_from_metadata(metadata: Dict[str, Any]) -> int:
    """Extract document year from metadata structures (source_pdf or chunk_id)."""
    source_pdf = metadata.get("source_pdf", "")
    match = re.search(r"\b(19\d{2}|20\d{2})\b", source_pdf)
    if match:
        return int(match.group(1))

    chunk_id = metadata.get("chunk_id", "")
    match = re.search(r"_(19\d{2}|20\d{2})_", chunk_id)
    if match:
        return int(match.group(1))

    return 2024

def adjust_similarity_scores(
    results: List[Dict[str, Any]],
    query_year: int = 2026,
    decay_lambda: float = 0.005
) -> List[Dict[str, Any]]:
    """
    Applies a linear decay penalty to similarity scores based on precedent age.
    Formula: Adjusted_Score = Cosine_Similarity - (decay_lambda * delta_years)
    """
    for r in results:
        doc_year = extract_year_from_metadata(r)
        delta_years = max(0, query_year - doc_year)
        cosine_sim = r.get("cosine_similarity", 0.0)
        adjusted_score = cosine_sim - (decay_lambda * delta_years)
        r["cosine_similarity"] = float(adjusted_score)

    results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results
