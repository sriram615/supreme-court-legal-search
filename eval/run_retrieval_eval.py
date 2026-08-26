"""
eval/run_retrieval_eval.py
==========================
Runs comprehensive retrieval-quality evaluation for the Indian Supreme Court Legal Search Engine.
Measures Precision@5, Hit@5 (Recall@5), and MRR@5 across 'faiss', 'bm25', and 'hybrid' modes
for all 120 queries in data/eval_set.jsonl.

Dumps full per-query evidence to eval/raw_results.jsonl.
Executes programmatic PASS/FAIL assertions comparing hybrid against single modes.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from search_engine import LegalSearchEngine


def mean(numbers: List[float]) -> float:
    return sum(numbers) / len(numbers) if numbers else 0.0


def run_evaluation(
    eval_set_path: str = "data/eval_set.jsonl",
    raw_output_path: str = "eval/raw_results.jsonl",
) -> Dict[str, Any]:
    print("=" * 70)
    print("  SC LEGAL SEARCH ENGINE RETRIEVAL EVALUATION")
    print("=" * 70)

    # 1. Load engine and index
    t0 = time.time()
    engine = LegalSearchEngine()
    engine.load_index()
    print(f"[+] Search engine loaded in {time.time() - t0:.2f}s. Total vectors: {engine.index.ntotal}")

    # 2. Load eval dataset
    if not os.path.exists(eval_set_path):
        raise FileNotFoundError(f"Evaluation set not found at: {eval_set_path}")

    with open(eval_set_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"[+] Loaded {len(queries)} ground-truth queries from {eval_set_path}")

    raw_results = []
    summary = {
        "faiss": {"p5": [], "hit5": [], "mrr": []},
        "bm25": {"p5": [], "hit5": [], "mrr": []},
        "hybrid": {"p5": [], "hit5": [], "mrr": []},
    }

    by_intent_summary = {}

    # 3. Evaluate each query in all 3 modes
    for idx, q_item in enumerate(queries, start=1):
        qid = q_item["query_id"]
        query = q_item["query"]
        itype = q_item["intent_type"]
        gt_pid = q_item["ground_truth"]

        if itype not in by_intent_summary:
            by_intent_summary[itype] = {
                "faiss": {"p5": [], "hit5": [], "mrr": []},
                "bm25": {"p5": [], "hit5": [], "mrr": []},
                "hybrid": {"p5": [], "hit5": [], "mrr": []},
            }

        q_raw = {
            "query_id": qid,
            "query": query,
            "intent_type": itype,
            "ground_truth": gt_pid,
            "modes": {},
        }

        for mode in ["faiss", "bm25", "hybrid"]:
            t_start = time.perf_counter()
            results = engine.search(query, top_k=5, mode=mode)
            latency_ms = (time.perf_counter() - t_start) * 1000

            retrieved_chunks = []
            hits_count = 0
            first_match_rank = 0

            for r_idx, hit in enumerate(results, start=1):
                hit_pid = hit.get("parent_judgment_id", "")
                is_match = (hit_pid == gt_pid)
                if is_match:
                    hits_count += 1
                    if first_match_rank == 0:
                        first_match_rank = r_idx

                score_val = hit.get(
                    "cosine_similarity" if mode == "faiss" else ("bm25_score" if mode == "bm25" else "rrf_score"),
                    0.0,
                )
                retrieved_chunks.append({
                    "rank": r_idx,
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_judgment_id": hit_pid,
                    "score": float(score_val),
                    "is_match": is_match,
                    "source_pdf": hit.get("source_pdf", ""),
                })

            p5 = hits_count / 5.0
            hit5 = 1.0 if hits_count > 0 else 0.0
            mrr = (1.0 / first_match_rank) if first_match_rank > 0 else 0.0

            summary[mode]["p5"].append(p5)
            summary[mode]["hit5"].append(hit5)
            summary[mode]["mrr"].append(mrr)

            by_intent_summary[itype][mode]["p5"].append(p5)
            by_intent_summary[itype][mode]["hit5"].append(hit5)
            by_intent_summary[itype][mode]["mrr"].append(mrr)

            q_raw["modes"][mode] = {
                "p5": p5,
                "hit5": hit5,
                "mrr": mrr,
                "latency_ms": latency_ms,
                "retrieved_chunks": retrieved_chunks,
            }

        raw_results.append(q_raw)

    # 4. Dump raw evidence
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for item in raw_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Dumped full raw evaluation evidence to {raw_output_path}")

    # 5. Print summary tables
    print("\n" + "=" * 70)
    print(f"OVERALL RETRIEVAL METRICS (n={len(queries)})")
    print("=" * 70)
    print(f"{'Mode':<10} | {'Precision@5':<12} | {'Hit@5 (Recall)':<15} | {'MRR@5':<10}")
    print("-" * 55)
    overall_avgs = {}
    for mode in ["faiss", "bm25", "hybrid"]:
        p5_avg = mean(summary[mode]["p5"])
        h5_avg = mean(summary[mode]["hit5"])
        mrr_avg = mean(summary[mode]["mrr"])
        overall_avgs[mode] = {"p5": p5_avg, "hit5": h5_avg, "mrr": mrr_avg}
        print(f"{mode:<10} | {p5_avg:<12.4f} | {h5_avg:<15.4f} | {mrr_avg:<10.4f}")

    print("\n" + "=" * 70)
    print("METRICS BY INTENT TYPE")
    print("=" * 70)
    intent_avgs = {}
    for itype in ["CITATION_OR_CASE_NO", "CASE_TITLE", "STATUTORY_SECTION", "CONCEPTUAL_OR_AMBIGUOUS"]:
        n_cat = len(by_intent_summary[itype]["faiss"]["p5"])
        print(f"\n--- Intent: {itype} (n={n_cat}) ---")
        print(f"{'Mode':<10} | {'Precision@5':<12} | {'Hit@5 (Recall)':<15} | {'MRR@5':<10}")
        print("-" * 55)
        intent_avgs[itype] = {}
        for mode in ["faiss", "bm25", "hybrid"]:
            p5_avg = mean(by_intent_summary[itype][mode]["p5"])
            h5_avg = mean(by_intent_summary[itype][mode]["hit5"])
            mrr_avg = mean(by_intent_summary[itype][mode]["mrr"])
            intent_avgs[itype][mode] = {"p5": p5_avg, "hit5": h5_avg, "mrr": mrr_avg}
            print(f"{mode:<10} | {p5_avg:<12.4f} | {h5_avg:<15.4f} | {mrr_avg:<10.4f}")

    # 6. Programmatic Pass/Fail Assertions (Rule #1 verification discipline)
    print("\n" + "=" * 70)
    print("PROGRAMMATIC ASSERTION CHECKS (HYBRID VS SINGLE MODES)")
    print("=" * 70)
    assertions = []

    # Check 1: Overall Hybrid Hit@5 >= FAISS & BM25
    h_h5 = overall_avgs["hybrid"]["hit5"]
    b_h5 = overall_avgs["bm25"]["hit5"]
    f_h5 = overall_avgs["faiss"]["hit5"]
    c1_pass = (h_h5 >= b_h5) and (h_h5 >= f_h5)
    assertions.append(("Overall Hybrid Hit@5 >= single modes", c1_pass, f"Hybrid: {h_h5:.4f} vs BM25: {b_h5:.4f}, FAISS: {f_h5:.4f}"))

    # Check 2: Overall Hybrid MRR >= FAISS & BM25
    h_mrr = overall_avgs["hybrid"]["mrr"]
    b_mrr = overall_avgs["bm25"]["mrr"]
    f_mrr = overall_avgs["faiss"]["mrr"]
    c2_pass = (h_mrr >= b_mrr) and (h_mrr >= f_mrr)
    assertions.append(("Overall Hybrid MRR >= single modes", c2_pass, f"Hybrid: {h_mrr:.4f} vs BM25: {b_mrr:.4f}, FAISS: {f_mrr:.4f}"))

    # Check 3: Hybrid does not underperform both single modes on ANY intent category
    all_cat_pass = True
    for itype in ["CITATION_OR_CASE_NO", "CASE_TITLE", "STATUTORY_SECTION", "CONCEPTUAL_OR_AMBIGUOUS"]:
        cat_h = intent_avgs[itype]["hybrid"]["hit5"]
        cat_b = intent_avgs[itype]["bm25"]["hit5"]
        cat_f = intent_avgs[itype]["faiss"]["hit5"]
        min_single = min(cat_b, cat_f)
        pass_cat = (cat_h >= min_single)
        if not pass_cat:
            all_cat_pass = False
        assertions.append((f"Intent {itype} Hybrid Hit@5 >= min(BM25, FAISS)", pass_cat, f"Hybrid: {cat_h:.4f} vs BM25: {cat_b:.4f}, FAISS: {cat_f:.4f}"))

    for name, status, detail in assertions:
        flag = "[PASS]" if status else "[FAIL]"
        print(f"{flag:<7} | {name:<50} | {detail}")

    all_passed = all(a[1] for a in assertions)
    print("-" * 70)
    print(f"FINAL ASSERTION SUITE RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)

    if not all_passed:
        raise AssertionError("Retrieval evaluation assertion failed: Hybrid underperformed single modes.")

    return {
        "summary": summary,
        "by_intent": by_intent_summary,
        "raw_results": raw_results,
    }


if __name__ == "__main__":
    run_evaluation()
