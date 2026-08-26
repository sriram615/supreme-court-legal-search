"""
tests/test_utils.py
===================
Tests for utils.py — covering every citation pattern, make_token normalization,
and adjust_similarity_scores decay math.

Negative cases are always realistic near-misses (e.g. wrong reporter, missing
vol, year out of range) rather than obviously-wrong strings, so they validate
the pattern's precision, not just its existence.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import _CITATION_PATTERNS, make_token, adjust_similarity_scores


# ─────────────────────────────────────────────────────────────────────────────
# 1. _CITATION_PATTERNS — 16 patterns, one positive + one near-miss each
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (pattern_key, positive_string, negative_string, neg_description)
_PATTERN_CASES = [
    # AIR: "AIR <year> <state> <page>"
    (
        "AIR",
        "AIR 2024 SC 100",
        "AIR 24 SC 100",           # year must be 4 digits
        "2-digit year rejected",
    ),
    # SCR: "(<year>) <vol> SCR <page>"
    (
        "SCR",
        "(2024) 3 SCR 250",
        "(24) 3 SCR 250",          # year must be 4 digits inside parens
        "2-digit year in parens rejected",
    ),
    # SCC: "(<year>) <vol> SCC <page>" or with Supp
    (
        "SCC",
        "(2024) 9 SCC 1",
        "(2024) SCC 1",            # vol number required between year and SCC
        "missing volume number rejected",
    ),
    # SCALE
    (
        "SCALE",
        "2024 (3) SCALE 512",
        "2024 CALE 512",           # SCALE must be spelled correctly
        "misspelled reporter rejected",
    ),
    # ILR
    (
        "ILR",
        "ILR (2023) 1 Delhi 200",
        "ILR 2023 SomeCourt 200",  # court name must be from allowed list
        "unlisted court name rejected",
    ),
    # JT
    (
        "JT",
        "JT 2024 (1) SC 99",
        "JT 2024 1 SC 99",         # vol must be in parens: (1)
        "volume without parens rejected",
    ),
    # SCWR
    (
        "SCWR",
        "2024 SCWR 135",
        "2024 SCWR",               # page number required after reporter
        "missing page number rejected",
    ),
    # MLJ
    (
        "MLJ",
        "2024 (2) MLJ 300",
        "2024 MLJ 300",            # volume in parens is required
        "missing parens volume rejected",
    ),
    # CLT
    (
        "CLT",
        "2023 (1) CLT 400",
        "2023 1 CLT 400",          # parens required around volume
        "volume without parens rejected",
    ),
    # GLH
    (
        "GLH",
        "2022 (3) GLH 75",
        "2022 (3) GHL 75",         # must be GLH not GHL
        "transposed letters rejected",
    ),
    # PLR
    (
        "PLR",
        "2024 (1) PLR 88",
        "PLR 2024 1 88",           # year must come first
        "year not first rejected",
    ),
    # BLJR
    (
        "BLJR",
        "2023 BLJR 44",
        "2023 BLJR",               # page number required
        "missing page rejected",
    ),
    # NLJ
    (
        "NLJ",
        "NLJ 2024 SC 55",
        "NLJ SC 55",               # year required between NLJ and SC
        "missing year rejected",
    ),
    # INSC
    (
        "INSC",
        "2024 INSC 762",
        "2024 INSC",               # page/number required after INSC
        "missing number after INSC rejected",
    ),
    # SLP
    (
        "SLP",
        "SLP (Civil) No. 12326 of 2024",
        "SLP No. 123 of 2024",     # number must be ≥ 1 digit — actually passes, check near-miss
        "SLP No. abc of 2024",     # non-numeric case number
        # overwrite: let's use a real near-miss
    ),
    # WRIT
    (
        "WRIT",
        "Writ Petition (Civil) No. 500 of 2024",
        "Writ Application No. 500 of 2024",   # must say "Petition" not "Application"
        "wrong keyword rejected",
    ),
]

# SLP near-miss fix: use a properly structured negative
_PATTERN_CASES[14] = (
    "SLP",
    "SLP (Civil) No. 12326 of 2024",
    "SLP No. xyz of 2024",         # non-numeric case number
    "non-numeric case number rejected",
)


@pytest.mark.parametrize(
    "key,positive,negative,neg_label",
    [
        (key, pos, neg, label)
        for key, pos, neg, label in _PATTERN_CASES
    ],
    ids=[f"{k}_positive" for k, *_ in _PATTERN_CASES] + [],  # pytest IDs auto-generated
)
def test_citation_pattern_positive(key, positive, negative, neg_label):
    """Pattern must match the canonical positive example."""
    pat = _CITATION_PATTERNS[key]
    assert pat.search(positive) is not None, (
        f"_CITATION_PATTERNS[{key!r}] failed to match positive: {positive!r}"
    )


@pytest.mark.parametrize(
    "key,positive,negative,neg_label",
    _PATTERN_CASES,
    ids=[f"{k}_negative" for k, *_ in _PATTERN_CASES],
)
def test_citation_pattern_negative(key, positive, negative, neg_label):
    """Pattern must NOT match the realistic near-miss string."""
    pat = _CITATION_PATTERNS[key]
    assert pat.search(negative) is None, (
        f"_CITATION_PATTERNS[{key!r}] incorrectly matched near-miss: {negative!r} "
        f"({neg_label})"
    )


def test_all_16_patterns_defined():
    """Guard: exactly 16 patterns defined — test suite stays in sync with code."""
    assert len(_CITATION_PATTERNS) == 16, (
        f"Expected 16 patterns, found {len(_CITATION_PATTERNS)}: {list(_CITATION_PATTERNS)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. make_token() normalization
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AIR 2024 SC 100",         "AIR_2024_SC_100"),
        ("(2024) 9 SCC 1",          "2024_9_SCC_1"),       # parens stripped
        ("2024 INSC 762",           "2024_INSC_762"),
        ("SLP (Civil) No. 12326 of 2024", "SLP_CIVIL_NO._12326_OF_2024"),
        ("JT 2024 (1) SC 99",       "JT_2024_1_SC_99"),    # parens collapsed
        ("  AIR  2024  SC  100  ",  "AIR_2024_SC_100"),    # leading/trailing stripped
        ("2024 (3) SCALE 512",      "2024_3_SCALE_512"),
    ],
)
def test_make_token_normalization(raw, expected):
    assert make_token(raw) == expected, (
        f"make_token({raw!r}) = {make_token(raw)!r}, expected {expected!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. adjust_similarity_scores() — decay math
# ─────────────────────────────────────────────────────────────────────────────

def test_adjust_similarity_scores_zero_delta():
    """
    Same year as query year: no decay.

    NOTE: extract_year_from_metadata uses r"\\b(19\\d{2}|20\\d{2})\\b" on source_pdf.
    Because underscore is a \\w char, "2026_SC.pdf" does NOT match (no \\b after '6').
    Use a filename where the year is at a true word boundary, e.g. "2026.pdf".
    """
    results = [{"source_pdf": "2026.pdf", "cosine_similarity": 0.80}]
    out = adjust_similarity_scores(results, query_year=2026, decay_lambda=0.005)
    assert out[0]["cosine_similarity"] == pytest.approx(0.80, abs=1e-9)


def test_adjust_similarity_scores_decay_10_years():
    """10-year-old precedent: score reduced by 0.005 * 10 = 0.05.

    Use a filename where '2016' sits at a word boundary: '2016.pdf'
    (digit string starting at the beginning of the filename).
    """
    results = [{"source_pdf": "2016.pdf", "cosine_similarity": 0.80}]
    out = adjust_similarity_scores(results, query_year=2026, decay_lambda=0.005)
    assert out[0]["cosine_similarity"] == pytest.approx(0.75, abs=1e-9)


def test_adjust_similarity_scores_future_doc_no_penalty():
    """Document year > query year: delta clamped to 0, no decay.

    Use chunk_id path because '2030_SC.pdf' would not match (underscore blocks \\b).
    """
    results = [{"source_pdf": "no_year.pdf", "chunk_id": "case_2030_chunk_01",
                "cosine_similarity": 0.60}]
    out = adjust_similarity_scores(results, query_year=2026, decay_lambda=0.005)
    assert out[0]["cosine_similarity"] == pytest.approx(0.60, abs=1e-9)


def test_adjust_similarity_scores_ranking_order():
    """Older doc should rank lower than newer doc with same raw similarity.

    Use chunk_id to guarantee year extraction (underscore pattern _YYYY_ is matched).
    2024: decay = 0.005*2 = 0.01 → 0.69
    2000: decay = 0.005*26 = 0.13 → 0.57
    """
    results = [
        {"source_pdf": "no_year.pdf", "chunk_id": "case_2000_chunk_01", "cosine_similarity": 0.70},
        {"source_pdf": "no_year.pdf", "chunk_id": "case_2024_chunk_01", "cosine_similarity": 0.70},
    ]
    out = adjust_similarity_scores(results, query_year=2026, decay_lambda=0.005)
    # 2024 doc: 0.70 - 0.005*2 = 0.69; 2000 doc: 0.70 - 0.005*26 = 0.57
    assert out[0]["chunk_id"] == "case_2024_chunk_01"
    assert out[1]["chunk_id"] == "case_2000_chunk_01"
    assert out[0]["rank"] == 1
    assert out[1]["rank"] == 2


def test_adjust_similarity_scores_chunk_id_fallback():
    """Year extracted from chunk_id when source_pdf has none."""
    results = [{"source_pdf": "unknown.pdf", "chunk_id": "ruling_2018_chunk_04", "cosine_similarity": 0.50}]
    out = adjust_similarity_scores(results, query_year=2026, decay_lambda=0.005)
    # delta = 2026 - 2018 = 8; decay = 0.04
    assert out[0]["cosine_similarity"] == pytest.approx(0.46, abs=1e-9)


def test_adjust_similarity_scores_no_year_defaults_2024():
    """No year anywhere: defaults to 2024."""
    results = [{"source_pdf": "no_year_here.pdf", "cosine_similarity": 0.50}]
    out = adjust_similarity_scores(results, query_year=2026, decay_lambda=0.005)
    # delta = 2026 - 2024 = 2; decay = 0.01
    assert out[0]["cosine_similarity"] == pytest.approx(0.49, abs=1e-9)
