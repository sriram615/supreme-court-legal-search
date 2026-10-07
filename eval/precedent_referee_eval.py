"""
precedent_referee_eval.py -- a REAL labeled evaluation of verify_contradiction_async(),
replacing run_benchmark_matrix()'s self-comparison benchmark for resume/interview purposes.

Why this exists (read before trusting any number it prints):
  validator.py's run_benchmark_matrix() calls verify_contradiction_async(text, text) --
  the SAME text on both sides -- and its "precision/recall/F1" are actually measuring
  agreement between the Legal-BERT classifier's prediction and the Ollama referee's
  prediction on that single input. That is an inter-model agreement rate, not accuracy
  against any ground truth, and it is not a number you can defend if asked "how do you
  know the referee actually works."

  This script instead evaluates verify_contradiction_async(argument_text, precedent_text)
  against a hand-reviewed labeled set (precedent_contradiction_labels.jsonl) of REAL pairs
  pulled from the actual corpus: 10 positive pairs mined from chunks where the corpus text
  itself uses explicit overruling language ("overruled", "no longer good law", "per
  incuriam", etc.) confirming a genuine contradiction, and 10 negative pairs of headnotes
  from clearly unrelated matters. See that file's "provenance" and "review_needed" fields --
  two of the ten positive pairs are flagged review_needed: true because the argument/
  precedent split required light paraphrasing of a single corpus sentence rather than
  being two independently-occurring verbatim spans. Skim those two before trusting the
  final number; the other eighteen are verbatim corpus text.

  20 pairs is a small sample -- big enough to compute a real, honest precision/recall/F1
  and to say something in an interview ("evaluated on a 20-pair hand-labeled sample drawn
  from real corpus citations, precision X, recall Y"), not big enough to claim statistical
  confidence. Say it that way. Extend the labels file with more pairs (same schema) if you
  want a tighter estimate later -- the mining approach in this file's sibling conversation
  (grep the corpus for overruling language, pair the old-rule chunk with the overruling
  chunk) generalizes; there were 308 corpus chunks containing that language, only 10 were
  used here.

Usage:
    ollama serve &                      # if not already running
    ollama pull phi3                    # if not already pulled
    .venv311/bin/python eval/precedent_referee_eval.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import verify_contradiction_async  # noqa: E402

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precedent_contradiction_labels.jsonl")


def load_labels(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


async def main():
    pairs = load_labels(LABELS_PATH)
    print(f"Loaded {len(pairs)} labeled pairs ({sum(p['label'] for p in pairs)} positive, "
          f"{sum(not p['label'] for p in pairs)} negative) from {LABELS_PATH}\n")

    tp = fp = tn = fn = 0
    offline_count = 0
    rows = []

    for p in pairs:
        verdict = await verify_contradiction_async(p["argument_text"], p["precedent_text"])
        pred = verdict.verdict_agreement
        truth = p["label"]

        if "Ollama LLM verification service is offline" in verdict.legal_rationale:
            offline_count += 1

        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and not truth:
            tn += 1
        else:
            fn += 1

        correct = "OK " if pred == truth else "OFF"
        rows.append(
            f"  [{correct}] {p['pair_id']:4s}  label={str(truth):5s}  pred={str(pred):5s}  "
            f"conf={verdict.confidence_rating:.2f}  {p['topic'][:60]}"
        )
        print(rows[-1])

    total = len(pairs)

    if offline_count == total:
        print(
            "\n*** Ollama never responded for ANY of the "
            f"{total} pairs (every verdict fell back to the offline default). "
            "These numbers are meaningless -- they reflect the fallback rule, not the "
            "referee. Start Ollama (`ollama serve`, `ollama pull phi3`) and re-run before "
            "trusting anything below. ***\n"
        )
    elif offline_count > 0:
        print(
            f"\n*** WARNING: {offline_count}/{total} calls hit the Ollama-offline fallback "
            "(timeout, connection refused, or bad JSON). Those pairs' predictions are the "
            "hardcoded default, not a real referee judgment -- treat the metrics below as "
            "optimistic and check the Ollama logs. ***\n"
        )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print("=" * 70)
    print("PRECEDENT-CONTRADICTION REFEREE -- LABELED EVALUATION (real ground truth)")
    print("=" * 70)
    print(f"Sample size      : {total} pairs (10 positive, 10 negative)")
    print(f"Confusion matrix : TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1               : {f1:.4f}")
    print(f"Accuracy         : {accuracy:.4f}")
    print("=" * 70)
    print(
        "\nHonest framing for a resume/interview: \"evaluated the precedent-conflict "
        f"referee against a 20-pair hand-labeled sample drawn from real corpus citations, "
        f"achieving {precision:.0%} precision / {recall:.0%} recall.\" Do not round this up "
        "to a general accuracy claim about the whole corpus -- the sample is real but small "
        "and was not randomly drawn (positives were selected for having explicit corpus "
        "language confirming the contradiction)."
    )


if __name__ == "__main__":
    asyncio.run(main())
