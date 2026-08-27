---
activation: always_on
---

# Verification Discipline

These rules apply to every task, especially performance fixes, index/cache/
lookup-table changes, and any change validated by a test you write yourself.

## 1. Don't widen logic just to pass your own test
If making a correctness test pass requires broadening a match, filter, or
regex, that is a signal to stop and check impact — not a green light. Before
reporting success, measure what the broadened logic now matches on realistic
inputs beyond the specific cases in the test.

## 2. Report distribution, not just aggregate counts
For any inverted index, cache, or lookup table you build: report the size of
the 5 largest buckets/keys, not just total key count or build time. Flag any
single key whose postings cover more than ~5% of the corpus — that key is not
functioning as a selective index and the fix is incomplete even if tests pass.

## 3. Your test set must include the common case, not just the reported case
When fixing a bug, test against the specific failing example AND against the
most frequent real-world variants of that input class (shortest values,
highest-frequency values, boundary lengths) — not only the examples from the
original bug report. A test set of 4-5 hand-picked queries is not sufficient
evidence for "100% correctness" claims.

## 4. Control for noise before making timing claims
Before reporting latency numbers or speedup multipliers: disable non-
deterministic external calls where possible (network lookups, cold caches),
run the measurement 3-5 times, and report the median. A single-run number is
not sufficient evidence for an "Nx faster" claim — say "median of N runs" or
don't claim a multiplier.

## 5. Self-check before declaring done
Before reporting a task complete, answer in one paragraph: "what realistic
input is most likely to break this fix that isn't in my test set?" Then
actually test that input. If you don't have time to test it, say so
explicitly in the report instead of omitting it.

## 6. State the actual strength of evidence
Never present a claim ("100% equivalence", "verified", "production-grade")
without the scope it was verified against. "Verified against 5 queries
covering 3-digit section numbers" is honest; "100% correctness" alone is not,
even if literally true for the cases tested.
