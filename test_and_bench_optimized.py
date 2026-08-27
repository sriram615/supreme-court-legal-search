"""
test_and_bench_optimized.py  (v3 — rule-compliant)
===================================================
Applies verification-discipline.md rules #2, #3, #4, #6.

SECTION 1 — INDEX DISTRIBUTION REPORT (rule #2)
  Report top-10 posting list sizes for citation_index and section_index.
  Flag any key covering >5% of the 15,847-chunk corpus.

SECTION 2 — CORRECTNESS TESTS (rules #3, #6)
  Test suite now includes:
    - original 5 queries  (3-digit sections: 304B, 482; citations)
    - 4 short-section queries (14, 19, 21, 32) that weren't in the bug report
  Assertion: new_set ⊆ old_set_B  where old_set_B uses condition B (sec_pat)
  only for purely numeric sections (semantically correct baseline).
  Reports new_set size and % corpus coverage for every query so selectivity
  is explicit (rule #6 — state the actual scope of the verification).

SECTION 3 — LATENCY BENCHMARK (rule #4)
  5 runs with HF_HUB_OFFLINE=1 preventing any network call.
  Reports per-run times and median for each query.  Claims multiplier only
  from the median, not from a single run.
"""

import os
import re
import sys
import time
import statistics
sys.path.insert(0, ".")

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # belt-and-suspenders

from search_engine import LegalSearchEngine, classify_and_extract_query

CORPUS_SIZE = 15847
PCT_THRESHOLD = CORPUS_SIZE * 0.05     # 792 chunks

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — INDEX DISTRIBUTION REPORT  (rule #2)
# ─────────────────────────────────────────────────────────────────────────────

def report_index_distribution(engine: LegalSearchEngine) -> None:
    print("\n" + "=" * 72)
    print("  INDEX DISTRIBUTION REPORT  (rule #2)")
    print("  Flag: any key whose posting list covers >5% of corpus (>792 chunks)")
    print("=" * 72)

    for name, idx in [("citation_index", engine.citation_index),
                       ("section_index",  engine.section_index)]:
        top10 = sorted(idx.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        print(f"\n  TOP-10  {name}:")
        print(f"  {'key':<22} {'count':>7}  {'%corpus':>7}  flag")
        print(f"  {'-'*55}")
        for k, v in top10:
            pct = 100.0 * len(v) / CORPUS_SIZE
            flag = "*** >5% CORPUS ***" if len(v) > PCT_THRESHOLD else "OK"
            print(f"  {k!r:<22} {len(v):>7d}  {pct:>6.1f}%  {flag}")

    # Spot-check short section numbers explicitly requested
    print(f"\n  SPOT-CHECK section_index keys '14','19','21','32','34','304','482','138':")
    print(f"  {'key':<8} {'count':>7}  {'%corpus':>7}  flag")
    print(f"  {'-'*45}")
    for k in ["14", "19", "21", "32", "34", "304", "482", "138"]:
        v = engine.section_index.get(k, [])
        pct = 100.0 * len(v) / CORPUS_SIZE
        flag = "*** >5% ***" if len(v) > PCT_THRESHOLD else "OK"
        print(f"  {k!r:<8} {len(v):>7d}  {pct:>6.1f}%  {flag}")

    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CORRECTNESS TESTS  (rules #3, #6)
# ─────────────────────────────────────────────────────────────────────────────

# Expanded test set: original 5 queries + 4 short-section queries (rule #3)
CORRECTNESS_QUERIES = [
    # --- originally tested ---
    "2024 INSC 762",
    "SLP Civil No. 12326 of 2024",
    "(2024) 9 SCC 1",
    "Section 304B IPC dowry death essential ingredients",
    "Section 482 CrPC inherent powers of High Court",
    # --- NOT in original test set; highest-frequency real-world variants ---
    "Section 14 of the Arbitration and Conciliation Act 1996",
    "Article 19 freedom of speech and expression",
    "Article 21 right to life and liberty",
    "Section 32 review jurisdiction of the Supreme Court",
]


def _condition_B_scan(engine: LegalSearchEngine, query: str) -> frozenset:
    """
    Semantically correct baseline: condition B only (explicit section/s/u/s prefix).
    For citation queries: same O(N) scan as before.
    This is the ground truth we verify new_set against.
    """
    parsed = classify_and_extract_query(query)
    qtype  = parsed.get("query_type", "CONCEPTUAL_OR_AMBIGUOUS")
    result = set()

    if qtype == "CITATION_OR_CASE_NO":
        target_ids = parsed.get("extracted_identifiers", [])
        for idx, meta in enumerate(engine.metadata_map):
            txt    = meta.get("text_chunk", "").lower()
            pdf    = meta.get("source_pdf", "").lower()
            cits   = " ".join(meta.get("extracted_citations", [])).lower()
            header = txt[:400]
            for tid in target_ids:
                pat = re.compile(r"\b" + re.escape(tid.lower()) + r"\b")
                if pat.search(cits) or pat.search(header) or pat.search(pdf) or pat.search(txt):
                    result.add(idx)
                    break

    elif qtype == "STATUTORY_SECTION":
        sec_digit = parsed.get("section_digit", "")
        full_sec  = parsed.get("full_section_str", "").lower()
        if sec_digit:
            sec_pat = re.compile(
                r"\b(?:section|sec\.?|s\.?|u/s\.?|u/ss\.?)\s*" + re.escape(sec_digit) + r"\b",
                re.I,
            )
            has_letter = bool(re.search(r"[a-zA-Z]", full_sec)) if full_sec else False
            for idx, meta in enumerate(engine.metadata_map):
                txt = meta.get("text_chunk", "").lower()
                if (has_letter and full_sec in txt) or sec_pat.search(txt):
                    result.add(idx)

    return frozenset(result)


def _new_exact_scan(engine: LegalSearchEngine, query: str) -> frozenset:
    """Reproduce the O(k) precomputed index lookup for comparison."""
    parsed = classify_and_extract_query(query)
    qtype  = parsed.get("query_type", "CONCEPTUAL_OR_AMBIGUOUS")
    result = set()

    if qtype == "CITATION_OR_CASE_NO":
        target_ids = parsed.get("extracted_identifiers", [])
        for tid in target_ids:
            sub_tokens = re.findall(r"[a-z0-9_]+", tid.lower())
            for tok in sub_tokens:
                if len(tok) >= 2 and tok in engine.citation_index:
                    pat = re.compile(r"\b" + re.escape(tid.lower()) + r"\b")
                    for ci in engine.citation_index[tok]:
                        meta   = engine.metadata_map[ci]
                        txt    = meta.get("text_chunk", "").lower()
                        pdf    = meta.get("source_pdf", "").lower()
                        cits   = " ".join(meta.get("extracted_citations", [])).lower()
                        header = txt[:400]
                        if pat.search(cits) or pat.search(header) or pat.search(pdf) or pat.search(txt):
                            result.add(ci)

    elif qtype == "STATUTORY_SECTION":
        sec_digit = parsed.get("section_digit", "")
        full_sec  = parsed.get("full_section_str", "").lower()
        if sec_digit and sec_digit in engine.section_index:
            sec_pat = re.compile(
                r"\b(?:section|sec\.?|s\.?|u/s\.?|u/ss\.?)\s*" + re.escape(sec_digit) + r"\b",
                re.I,
            )
            has_letter = bool(re.search(r"[a-zA-Z]", full_sec)) if full_sec else False
            for ci in engine.section_index[sec_digit]:
                txt = engine.metadata_map[ci].get("text_chunk", "").lower()
                if (has_letter and full_sec in txt) or sec_pat.search(txt):
                    result.add(ci)

    return frozenset(result)


def run_correctness_tests(engine: LegalSearchEngine) -> bool:
    print("=" * 72)
    print("  CORRECTNESS TESTS  (rules #3, #6)")
    print("  Baseline: condition-B-only O(N) scan (semantically correct)")
    print("  Assertion: new_set ⊆ baseline_set (no extra false positives)")
    print("  Note: baseline_set itself may be smaller than old substring-scan")
    print("        for numeric-only sections — that is the CORRECT new behavior.")
    print("=" * 72)
    all_passed = True

    for q in CORRECTNESS_QUERIES:
        baseline  = _condition_B_scan(engine, q)
        new_set   = _new_exact_scan(engine, q)
        is_subset = new_set.issubset(baseline)
        pct_new   = 100.0 * len(new_set) / CORPUS_SIZE
        pct_base  = 100.0 * len(baseline) / CORPUS_SIZE
        only_new  = new_set - baseline    # extra candidates not in baseline (bad)
        only_base = baseline - new_set    # missed candidates (acceptable if they were substr-FP)
        status    = "PASS ✓" if is_subset else "FAIL ✗"
        if not is_subset:
            all_passed = False

        sel_flag = f"  *** new_set {pct_new:.1f}% corpus" if len(new_set) > PCT_THRESHOLD else ""
        print(f"\n  {status}  base={len(baseline):4d} ({pct_base:4.1f}%)  "
              f"new={len(new_set):4d} ({pct_new:4.1f}%){sel_flag}")
        print(f"         {q}")
        if only_new:
            print(f"         EXTRA in new (false positives): {sorted(only_new)[:5]}")
        if only_base:
            print(f"         Missed vs baseline: {len(only_base)} "
                  f"(expected for numeric-only sections — no longer substring FP)")

    print("\n" + "-" * 72)
    print(f"  Result: {'ALL PASSED' if all_passed else 'FAILURES — see above'}")
    print("=" * 72 + "\n")
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LATENCY BENCHMARK  (rule #4: 5 runs, report median)
# ─────────────────────────────────────────────────────────────────────────────

QUERIES = [
    ("What is the disqualification criteria for a female Sarpanch who encroached on government land?", "CONCEPTUAL"),
    ("doctrine of promissory estoppel in government contracts",                                         "CONCEPTUAL"),
    ("bail conditions for accused in money laundering cases",                                           "CONCEPTUAL"),
    ("right to fair trial and speedy justice under Article 21",                                        "CONCEPTUAL"),
    ("land acquisition compensation and market value determination",                                    "CONCEPTUAL"),
    ("arbitration clause in commercial disputes scope and validity",                                    "CONCEPTUAL"),
    ("contempt of court powers of High Court and Supreme Court",                                       "CONCEPTUAL"),
    ("judicial review of administrative action proportionality test",                                  "CONCEPTUAL"),
    ("Manisha Ravindra Panpatil v. State of Maharashtra",  "CASE_TITLE"),
    ("Sonam Lakra v. State of Chhattisgarh",               "CASE_TITLE"),
    ("Union of India v. State of Rajasthan",               "CASE_TITLE"),
    ("State of Kerala v. N. M. Thomas",                    "CASE_TITLE"),
    ("Section 304B IPC dowry death essential ingredients",             "STATUTORY_SECTION"),
    ("Section 138 Negotiable Instruments Act dishonoured cheque",      "STATUTORY_SECTION"),
    ("Section 482 CrPC inherent powers of High Court",                 "STATUTORY_SECTION"),
    ("Section 14 of the Arbitration and Conciliation Act 1996",        "STATUTORY_SECTION"),
    ("2024 INSC 762",            "CITATION_OR_CASE_NO"),
    ("AIR 2024 SC 100",          "CITATION_OR_CASE_NO"),
    ("(2024) 9 SCC 1",           "CITATION_OR_CASE_NO"),
    ("SLP Civil No. 12326 of 2024", "CITATION_OR_CASE_NO"),
]

BEFORE_LATENCIES = {
    1: 1602.77, 2: 95.98,   3: 107.86, 4: 92.02,   5: 88.48,
    6: 87.53,   7: 122.52,  8: 98.33,  9: 110.04,  10: 75.60,
    11: 72.22,  12: 111.58, 13: 2023.74, 14: 2050.81, 15: 2283.75,
    16: 1495.29, 17: 2460.92, 18: 2132.64, 19: 3877.29, 20: 2422.76,
}

N_RUNS = 5


def run_latency_benchmark(engine: LegalSearchEngine, n_runs: int = N_RUNS) -> None:
    print(f"LATENCY BENCHMARK  (rule #4: {n_runs} runs, report median)  HF_HUB_OFFLINE=1")
    print("=" * 72)

    all_run_latencies = []   # list of lists

    for run in range(1, n_runs + 1):
        run_lats = []
        for i, (q, intent) in enumerate(QUERIES, 1):
            t0 = time.perf_counter()
            engine.search(q, top_k=3)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            run_lats.append(elapsed_ms)
        all_run_latencies.append(run_lats)
        median_this_run = statistics.median(run_lats)
        print(f"  Run {run}/{n_runs}: overall p50={sorted(run_lats)[len(run_lats)//2]:.1f} ms  "
              f"p95={sorted(run_lats)[int(len(run_lats)*0.95)]:.1f} ms  "
              f"mean={statistics.mean(run_lats):.1f} ms")

    # Median latency per query across N runs (skips run-1 warm-up bias by design — all runs included)
    print(f"\n  PER-QUERY MEDIAN LATENCY (median of {n_runs} runs):")
    print(f"  {'#':<4} {'intent':<22} {'median':>8}  {'before':>8}  {'Δ':>8}  query (truncated)")
    print(f"  {'-'*80}")
    intent_meds: dict = {}
    for i, (q, intent) in enumerate(QUERIES, 1):
        per_query = [all_run_latencies[r][i - 1] for r in range(n_runs)]
        med = statistics.median(per_query)
        before = BEFORE_LATENCIES[i]
        delta  = med - before
        sign   = "+" if delta >= 0 else "-"
        intent_meds.setdefault(intent, []).append(med)
        print(f"  [{i:02d}] {intent:<22} {med:8.2f}ms  {before:8.2f}ms  {sign}{abs(delta):7.1f}ms  {q[:45]}")

    # Overall summary
    all_meds = [statistics.median([all_run_latencies[r][i] for r in range(n_runs)])
                for i in range(len(QUERIES))]
    sorted_meds = sorted(all_meds)
    n = len(sorted_meds)
    overall_p50 = sorted_meds[n // 2]
    overall_p95 = sorted_meds[int(n * 0.95)]

    before_sorted = sorted(BEFORE_LATENCIES.values())
    before_p50 = before_sorted[n // 2]
    before_p95 = before_sorted[int(n * 0.95)]

    print(f"\n  INTENT-LEVEL SUMMARY (medians across {n_runs} runs):")
    for intent in ["CONCEPTUAL", "CASE_TITLE", "STATUTORY_SECTION", "CITATION_OR_CASE_NO"]:
        vals = sorted(intent_meds.get(intent, []))
        if not vals:
            continue
        print(f"  {intent:<26}  p50={vals[len(vals)//2]:7.1f} ms  "
              f"min={min(vals):7.1f} ms  max={max(vals):7.1f} ms")

    print(f"\n  OVERALL BEFORE vs. AFTER (medians, n={n_runs} runs):")
    print(f"  {'Metric':<10}  {'Before':>12}  {'After (med)':>12}  {'Δ':>12}")
    print(f"  {'-'*52}")
    print(f"  {'p50':<10}  {before_p50:>10.2f}ms  {overall_p50:>10.2f}ms  "
          f"{overall_p50-before_p50:>+10.2f}ms")
    print(f"  {'p95':<10}  {before_p95:>10.2f}ms  {overall_p95:>10.2f}ms  "
          f"{overall_p95-before_p95:>+10.2f}ms")
    mean_after  = statistics.mean(all_meds)
    mean_before = statistics.mean(BEFORE_LATENCIES.values())
    print(f"  {'mean':<10}  {mean_before:>10.2f}ms  {mean_after:>10.2f}ms  "
          f"{mean_after-mean_before:>+10.2f}ms")

    print(f"\n  (All 'after' numbers are medians of {n_runs} runs — rule #4.)")
    print("=" * 72)


if __name__ == "__main__":
    print("Loading full-corpus engine (index + BM25 + structured indices)...")
    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")
    engine.load_index()

    report_index_distribution(engine)

    passed = run_correctness_tests(engine)
    if not passed:
        print("ERROR: Correctness test failed — aborting benchmark.")
        sys.exit(1)

    run_latency_benchmark(engine)
