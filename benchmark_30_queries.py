#!/usr/bin/env python3
"""
benchmark_30_queries.py — Fires all 30 benchmark queries against the live
Legal Search Engine on http://127.0.0.1:8000/api/v1/analyze and writes
raw results to benchmark_results.json.
"""
import json, time, requests

API = "http://127.0.0.1:8000/api/v1/analyze"

QUERIES = [
    # ── A. Known-case fact-pattern queries (1-7) ──
    {"id": 1,  "section": "A", "query": "presumption of dowry death when a woman dies an unnatural death within seven years of marriage after being subjected to cruelty"},
    {"id": 2,  "section": "A", "query": "land acquisition compensation lapses after the government has already taken possession"},
    {"id": 3,  "section": "A", "query": "insolvency resolution application filed by a financial creditor under section 7"},
    {"id": 4,  "section": "A", "query": "enforceability of an arbitral award in a commercial dispute"},
    {"id": 5,  "section": "A", "query": "habeas corpus petition against illegal detention seeking bail under article 21"},
    {"id": 6,  "section": "A", "query": "appellate court's approach to evaluating conflicting witness evidence in a criminal appeal"},
    {"id": 7,  "section": "A", "query": "violation of principles of natural justice raised in a multi-judge bench procedural appeal"},

    # ── B. Exact / structured citation lookups (8-12) ──
    {"id": 8,  "section": "B", "query": "2024_INSC_762"},
    {"id": 9,  "section": "B", "query": "2024_INSC_737"},
    {"id": 10, "section": "B", "query": "[2024] 8 S.C.R. 798"},
    {"id": 11, "section": "B", "query": "2024_INSC_554"},
    {"id": 12, "section": "B", "query": "2024_INSC_9999"},

    # ── C. Domain-confirmed fact patterns (13-19) ──
    {"id": 13, "section": "C", "query": "anticipatory bail sought in a money laundering investigation"},
    {"id": 14, "section": "C", "query": "disqualification of a sarpanch for encroaching on panchayat land"},
    {"id": 15, "section": "C", "query": "anticipatory bail application under section 438 CrPC in a non-money-laundering criminal case"},
    {"id": 16, "section": "C", "query": "regular bail application filed under section 439 CrPC after the chargesheet was filed"},
    {"id": 17, "section": "C", "query": "quashing of an FIR using the inherent powers of the High Court under section 482 CrPC"},
    {"id": 18, "section": "C", "query": "abetment of suicide under section 306 of the Indian Penal Code"},
    {"id": 19, "section": "C", "query": "application for appointment of an arbitrator under section 11 of the Arbitration and Conciliation Act"},

    # ── D. Near-miss / adversarial precision tests (20-24) ──
    {"id": 20, "section": "D", "query": "cruelty by husband and in-laws under section 498A, no death involved"},
    {"id": 21, "section": "D", "query": "murder committed with common intention under sections 302 and 34 IPC"},
    {"id": 22, "section": "D", "query": "insolvency application filed by an operational creditor, not a financial creditor"},
    {"id": 23, "section": "D", "query": "challenge to an arbitral award specifically on the ground of patent illegality"},
    {"id": 24, "section": "D", "query": "encroachment on private agricultural land by an individual, no elected office involved"},

    # ── E. Negative controls (25-28) ──
    {"id": 25, "section": "E", "query": "trademark infringement dispute between two competing software companies"},
    {"id": 26, "section": "E", "query": "divorce and child custody dispute in a Hindu marriage"},
    {"id": 27, "section": "E", "query": "motor vehicle accident compensation claim under the Motor Vehicles Act"},
    {"id": 28, "section": "E", "query": "income tax dispute over a deemed dividend assessment"},

    # ── F. Statute-level / cross-citation searches (29-30) ──
    {"id": 29, "section": "F", "query": "cases discussing section 113B of the Evidence Act, presumption as to dowry death"},
    {"id": 30, "section": "F", "query": "judgments citing article 21 for speedy trial or personal liberty"},
]

results = []
for q in QUERIES:
    t0 = time.perf_counter()
    try:
        resp = requests.post(API, json={"text": q["query"]}, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        data = resp.json()
        matches = data.get("semantic_matches", [])
        top = matches[0] if matches else None

        result = {
            "id": q["id"],
            "section": q["section"],
            "query": q["query"],
            "latency_ms": round(elapsed, 1),
            "num_matches": len(matches),
            "top1_judgment_id": top.get("parent_judgment_id", "") if top else "NO_MATCH",
            "top1_chunk_id": top.get("chunk_id", "") if top else "NO_MATCH",
            "top1_source_pdf": top.get("source_pdf", "") if top else "NO_MATCH",
            "top1_court": top.get("court", "") if top else "",
            "top1_rrf_score": round(top.get("rrf_score", 0), 6) if top else 0,
            "top1_cosine": round(top.get("cosine_similarity", 0), 6) if top else 0,
            "top1_bm25": round(top.get("bm25_score", 0), 4) if top else 0,
            "top1_snippet": (top.get("text_chunk", "")[:300].replace("\n", " ")) if top else "",
            "top1_citations": top.get("extracted_citations", []) if top else [],
            "query_type": data.get("query_type", ""),
            "resolved_citations": data.get("resolved_citations", {}),
            "verification_verdict": data.get("verification_verdict", {}),
        }
        # Also capture top 2 and top 3 judgment IDs for context
        if len(matches) > 1:
            result["top2_judgment_id"] = matches[1].get("parent_judgment_id", "")
            result["top2_rrf_score"] = round(matches[1].get("rrf_score", 0), 6)
        if len(matches) > 2:
            result["top3_judgment_id"] = matches[2].get("parent_judgment_id", "")
            result["top3_rrf_score"] = round(matches[2].get("rrf_score", 0), 6)

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        result = {
            "id": q["id"],
            "section": q["section"],
            "query": q["query"],
            "latency_ms": round(elapsed, 1),
            "error": str(exc),
        }

    results.append(result)
    status = f"Q{result['id']:02d} [{result.get('query_type', '?')}] → {result.get('top1_judgment_id', 'ERR')} (RRF={result.get('top1_rrf_score', 0):.4f}, cos={result.get('top1_cosine', 0):.4f}) {result.get('latency_ms', 0):.0f}ms"
    print(status)

# Write full results
with open("benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ All {len(results)} queries completed. Results saved to benchmark_results.json")
