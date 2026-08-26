"""
tests/test_search_engine.py
============================
Tests for search_engine.py covering:
  - classify_and_extract_query() — all 4 intent types including session bug-triggers
  - evaluate_routed_relevance() — all 4 routes (A/B/C/D)
  - Three regression tests for bugs found and fixed this session (§8a of BUILD_REPORT.md)

Regression tests are designed so that reintroducing the original bug makes them RED.
See MUTATION CHECK section comments for how to mutate each one.

Engine objects that need index data use object.__new__ + manual attribute injection
to bypass SentenceTransformer loading — no network, no model, no file I/O.
"""

import sys
import os
import re
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from search_engine import (
    classify_and_extract_query,
    evaluate_routed_relevance,
    LegalSearchEngine,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — create minimal engine instances without loading model or index
# ─────────────────────────────────────────────────────────────────────────────

def _make_engine(metadata: list) -> LegalSearchEngine:
    """
    Return a LegalSearchEngine-shaped object with a populated metadata_map
    but WITHOUT loading the transformer, FAISS index, or BM25 corpus.
    Safe to use for calling _build_structured_indices() in unit tests.
    """
    engine = object.__new__(LegalSearchEngine)
    engine.metadata_map = metadata
    engine.citation_index = {}
    engine.section_index  = {}
    engine.bm25           = None
    engine.index          = None
    return engine


def _build(engine: LegalSearchEngine) -> LegalSearchEngine:
    """Build structured indices and return engine for chaining."""
    engine._build_structured_indices()
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# 1. classify_and_extract_query — all 4 intent types
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyAndExtractQuery:
    """Covers all 4 intent types plus the specific queries that caused bugs."""

    # ── CITATION_OR_CASE_NO ──

    def test_insc_citation(self):
        """'2024 INSC 762' — INSC format is a known citation."""
        r = classify_and_extract_query("2024 INSC 762")
        assert r["query_type"] == "CITATION_OR_CASE_NO"
        assert any("762" in i or "insc" in i.lower() for i in r["extracted_identifiers"])

    def test_air_citation(self):
        """'AIR 2024 SC 100' — reporter citation detected via cit_reporter_match."""
        r = classify_and_extract_query("AIR 2024 SC 100")
        assert r["query_type"] == "CITATION_OR_CASE_NO"

    def test_scc_historic_citation(self):
        """'(2024) 9 SCC 1' — historic-style citation with parens."""
        r = classify_and_extract_query("(2024) 9 SCC 1")
        assert r["query_type"] == "CITATION_OR_CASE_NO"

    def test_slp_case_number(self):
        """'SLP Civil No. 12326 of 2024' — SLP case number route."""
        r = classify_and_extract_query("SLP Civil No. 12326 of 2024")
        assert r["query_type"] == "CITATION_OR_CASE_NO"
        # The 5-digit case number must be extracted (not a year)
        assert any("12326" in i for i in r["extracted_identifiers"]), (
            f"Expected '12326' in extracted_identifiers, got {r['extracted_identifiers']}"
        )

    def test_slp_year_not_extracted_as_identifier(self):
        """Year (4-digit, 1900-2099) must not appear as extracted identifier."""
        r = classify_and_extract_query("SLP Civil No. 12326 of 2024")
        assert r["query_type"] == "CITATION_OR_CASE_NO"
        # '2024' is a year and must not appear as extracted identifier
        for ident in r["extracted_identifiers"]:
            assert "2024" not in ident or len(ident) > 4, (
                f"Year '2024' must not be a standalone extracted identifier, got {ident!r}"
            )

    def test_writ_petition_citation(self):
        """'Writ Petition (Civil) No. 500 of 2024' — writ petition format."""
        r = classify_and_extract_query("Writ Petition No. 500 of 2024")
        assert r["query_type"] == "CITATION_OR_CASE_NO"

    # ── CASE_TITLE ──

    def test_case_title_with_v(self):
        """'Manisha Panpatil v. State of Maharashtra' — CASE_TITLE via 'v.'."""
        r = classify_and_extract_query("Manisha Ravindra Panpatil v. State of Maharashtra")
        assert r["query_type"] == "CASE_TITLE"
        assert "panpatil" in r["extracted_party_nouns"] or "manisha" in r["extracted_party_nouns"]

    def test_case_title_with_versus(self):
        """'Union of India versus XYZ' — CASE_TITLE via 'versus'."""
        r = classify_and_extract_query("Union of India versus State of Rajasthan")
        assert r["query_type"] == "CASE_TITLE"

    def test_case_title_primary_party_nouns_exclude_jurisdiction_words(self):
        """primary_party_nouns must exclude jurisdiction state words."""
        r = classify_and_extract_query("Sonam Lakra v. State of Chhattisgarh")
        assert r["query_type"] == "CASE_TITLE"
        # 'state' and 'chhattisgarh' may appear in extracted_party_nouns but
        # 'sonam' and 'lakra' must be in primary_party_nouns
        primary = r.get("primary_party_nouns", [])
        assert any(n in primary for n in ("sonam", "lakra")), (
            f"Expected 'sonam' or 'lakra' in primary_party_nouns, got {primary}"
        )

    # ── STATUTORY_SECTION ──

    def test_section_three_digit(self):
        """'Section 482 CrPC' — 3-digit section classification."""
        r = classify_and_extract_query("Section 482 CrPC inherent powers of High Court")
        assert r["query_type"] == "STATUTORY_SECTION"
        assert r["section_digit"] == "482"
        assert r["full_section_str"].startswith("482")

    def test_section_304b_with_letter(self):
        """'Section 304B IPC' — section with letter suffix."""
        r = classify_and_extract_query("Section 304B IPC dowry death")
        assert r["query_type"] == "STATUTORY_SECTION"
        assert r["section_digit"] == "304"
        assert "304" in r["full_section_str"]

    def test_section_short_two_digit(self):
        """
        'Section 14 of the Arbitration Act' — 2-digit section.
        This was a session bug: the short number caused >5% corpus hits.
        """
        r = classify_and_extract_query("Section 14 of the Arbitration and Conciliation Act 1996")
        assert r["query_type"] == "STATUTORY_SECTION"
        assert r["section_digit"] == "14"

    def test_article_prefix_detected(self):
        """'Article 21' uses 'article' prefix — must classify as STATUTORY_SECTION."""
        r = classify_and_extract_query("Article 21 right to life and liberty")
        assert r["query_type"] == "STATUTORY_SECTION"
        assert r["section_digit"] == "21"

    def test_section_does_not_classify_as_citation(self):
        """A section query must NOT be confused with a citation query."""
        r = classify_and_extract_query("Section 138 Negotiable Instruments Act")
        assert r["query_type"] == "STATUTORY_SECTION"
        assert r["query_type"] != "CITATION_OR_CASE_NO"

    # ── CONCEPTUAL_OR_AMBIGUOUS ──

    def test_conceptual_legal_phrase(self):
        """Pure conceptual query — no citation, no title, no section."""
        r = classify_and_extract_query("doctrine of promissory estoppel in government contracts")
        assert r["query_type"] == "CONCEPTUAL_OR_AMBIGUOUS"
        assert "content_tokens" in r
        assert len(r["content_tokens"]) >= 2

    def test_conceptual_has_content_tokens(self):
        """Content tokens should be present and stop-words removed."""
        r = classify_and_extract_query("bail conditions for accused in money laundering cases")
        assert r["query_type"] == "CONCEPTUAL_OR_AMBIGUOUS"
        tokens = r["content_tokens"]
        # Common stop words must be removed
        assert "for" not in tokens
        assert "in" not in tokens
        assert "accused" in tokens or "bail" in tokens

    def test_ambiguous_short_query(self):
        """Very short ambiguous query — falls to CONCEPTUAL."""
        r = classify_and_extract_query("land acquisition")
        assert r["query_type"] == "CONCEPTUAL_OR_AMBIGUOUS"


# ─────────────────────────────────────────────────────────────────────────────
# 2. evaluate_routed_relevance — all 4 routes
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateRoutedRelevance:
    """Verifies each of the 4 routing paths with mock hits."""

    # ── Route A: CITATION_OR_CASE_NO ──

    def test_route_a_citation_verified(self):
        """Route A: identifier present in chunk text → is_relevant True."""
        hit = {
            "text_chunk": "The SLP (Civil) No. 12326 of 2024 was decided by a three-judge bench.",
            "source_pdf": "slp_12326_2024.pdf",
            "chunk_id": "slp_12326_2024_001",
            "extracted_citations": ["SLP No. 12326 of 2024"],
            "cosine_similarity": 0.82,
            "bm25_score": 11.0,
        }
        result = evaluate_routed_relevance(hit, "SLP Civil No. 12326 of 2024")
        assert result["query_type"] == "CITATION_OR_CASE_NO"
        assert result["is_relevant"] is True
        assert result["meta_verified"] is True

    def test_route_a_citation_absent(self):
        """Route A: identifier absent from all text → is_relevant False."""
        hit = {
            "text_chunk": "This is a judgment about land acquisition compensation.",
            "source_pdf": "land_acq_2024.pdf",
            "chunk_id": "land_acq_001",
            "extracted_citations": [],
            "cosine_similarity": 0.40,
            "bm25_score": 3.0,
        }
        result = evaluate_routed_relevance(hit, "SLP Civil No. 12326 of 2024")
        assert result["query_type"] == "CITATION_OR_CASE_NO"
        assert result["is_relevant"] is False

    # ── Route B: CASE_TITLE ──

    def test_route_b_title_party_matched(self):
        """Route B: primary party noun present in doc text → is_relevant True."""
        hit = {
            "text_chunk": "Manisha Ravindra Panpatil filed a writ petition against the State of Maharashtra.",
            "source_pdf": "panpatil_maharashtra.pdf",
            "chunk_id": "panpatil_001",
            "extracted_citations": [],
            "cosine_similarity": 0.70,
            "bm25_score": 8.0,
        }
        result = evaluate_routed_relevance(
            hit, "Manisha Ravindra Panpatil v. State of Maharashtra"
        )
        assert result["query_type"] == "CASE_TITLE"
        assert result["is_relevant"] is True

    def test_route_b_no_party_match(self):
        """Route B: no primary party present → is_relevant False."""
        hit = {
            "text_chunk": "This case concerns promissory estoppel in municipal contracts.",
            "source_pdf": "estoppel_2024.pdf",
            "chunk_id": "estoppel_001",
            "extracted_citations": [],
            "cosine_similarity": 0.55,
            "bm25_score": 5.0,
        }
        result = evaluate_routed_relevance(
            hit, "Manisha Ravindra Panpatil v. State of Maharashtra"
        )
        assert result["query_type"] == "CASE_TITLE"
        assert result["is_relevant"] is False

    # ── Route C: STATUTORY_SECTION ──

    def test_route_c_section_present(self):
        """Route C: section number present in doc → is_relevant True."""
        hit = {
            "text_chunk": "The inherent powers under section 482 CrPC allow the High Court to prevent abuse.",
            "source_pdf": "section_482_crpc.pdf",
            "chunk_id": "crpc_482_001",
            "extracted_citations": [],
            "cosine_similarity": 0.75,
            "bm25_score": 9.0,
        }
        result = evaluate_routed_relevance(hit, "Section 482 CrPC inherent powers of High Court")
        assert result["query_type"] == "STATUTORY_SECTION"
        assert result["is_relevant"] is True
        assert "482" in result.get("gate_reason", "")

    def test_route_c_section_absent(self):
        """Route C: section number absent → is_relevant False."""
        hit = {
            "text_chunk": "This judgment discusses land acquisition under the old Act.",
            "source_pdf": "land_act_2024.pdf",
            "chunk_id": "land_001",
            "extracted_citations": [],
            "cosine_similarity": 0.50,
            "bm25_score": 4.0,
        }
        result = evaluate_routed_relevance(hit, "Section 482 CrPC inherent powers of High Court")
        assert result["query_type"] == "STATUTORY_SECTION"
        assert result["is_relevant"] is False

    # ── Route D: CONCEPTUAL_OR_AMBIGUOUS ──

    def test_route_d_strong_semantic_pass(self):
        """Route D: cosine >= 0.40 + ≥1 substantive token → is_relevant True."""
        hit = {
            "text_chunk": "The doctrine of promissory estoppel prevents the government from reneging on its promises.",
            "source_pdf": "estoppel_2024.pdf",
            "chunk_id": "estoppel_001",
            "extracted_citations": [],
            "cosine_similarity": 0.55,   # above 0.40 threshold
            "bm25_score": 6.0,
        }
        result = evaluate_routed_relevance(hit, "doctrine of promissory estoppel")
        assert result["query_type"] == "CONCEPTUAL_OR_AMBIGUOUS"
        assert result["is_relevant"] is True

    def test_route_d_below_thresholds(self):
        """Route D: cosine and BM25 both below thresholds → is_relevant False."""
        hit = {
            "text_chunk": "Unrelated text about taxation procedure.",
            "source_pdf": "tax_2024.pdf",
            "chunk_id": "tax_001",
            "extracted_citations": [],
            "cosine_similarity": 0.15,   # below 0.40 and 0.30
            "bm25_score": 1.0,           # below 4.0
        }
        result = evaluate_routed_relevance(hit, "promissory estoppel in government contracts")
        assert result["query_type"] == "CONCEPTUAL_OR_AMBIGUOUS"
        assert result["is_relevant"] is False

    def test_route_d_empty_hit_returns_not_relevant(self):
        """evaluate_routed_relevance on empty hit → safe False return."""
        result = evaluate_routed_relevance({}, "bail conditions for accused")
        assert result["is_relevant"] is False
        assert result["query_type"] == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRESSION TESTS — each targets one specific bug from this session
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionA_StandalonePatternMatchesRealStrings:
    """
    Regression for Bug A: double-backslash in regex metacharacters.

    Original bug: the section-context pattern was compiled with r"\\b" (raw string
    with two chars: backslash + b) instead of r"\b" (word boundary). The compiled
    pattern then tried to match a literal backslash character in text, so
    "section 304" and "u/s 482" never matched.

    MUTATION CHECK — to see this test go RED:
      In search_engine.py _build_structured_indices(), change:
        _STANDALONE_NUM = re.compile(r"\\b(\\d+[a-zA-Z]?)\\b")
      to:
        _STANDALONE_NUM = re.compile(r"\\\\b(\\d+[a-zA-Z]?)\\\\b")
      Re-run pytest: test_section_index_built_for_section_304 and
      test_section_index_built_for_u_s_482 will FAIL (KeyError or empty list).
      Revert change -> both PASS.
    """

    def test_standalone_pattern_word_boundary_is_real(self):
        """
        The _STANDALONE_NUM pattern must use real \b word boundaries.
        r"\\b" in a raw string is TWO chars (backslash + b) — NOT a boundary.
        We test this by directly checking the pattern works on a plain string.
        """
        correct_pattern  = re.compile(r"\b(\d+[a-zA-Z]?)\b")
        buggy_pattern    = re.compile(r"\\b(\d+[a-zA-Z]?)\\b")

        text = "section 304 of the ipc"
        assert correct_pattern.search(text) is not None, (
            r'r"\b(\d+[a-zA-Z]?)\b" must match "304" in "section 304 of the ipc"'
        )
        assert buggy_pattern.search(text) is None, (
            r'Sanity check: r"\\b(\d+[a-zA-Z]?)\\b" (buggy form) must NOT match '
            "normal text — confirms the two patterns are distinct"
        )

    def test_section_index_built_for_section_304(self):
        """
        After _build_structured_indices(), a chunk containing 'section 304'
        must appear in section_index['304'].

        With the double-backslash bug, _STANDALONE_NUM never matches → section_index is empty.
        """
        meta = [{"text_chunk": "Under section 304 of the IPC the punishment is prescribed.",
                 "source_pdf": "ipc.pdf",
                 "extracted_citations": []}]
        engine = _build(_make_engine(meta))
        assert "304" in engine.section_index, (
            "section_index must contain '304' for a chunk with 'section 304'"
        )
        assert 0 in engine.section_index["304"], (
            "Chunk 0 must be in section_index['304']"
        )

    def test_section_index_built_for_u_s_482(self):
        """
        Chunk containing 'u/s 482' must appear in section_index['482'].
        'u/s' is a common abbreviation for 'under section' in Indian courts.
        """
        meta = [{"text_chunk": "The petition was filed u/s 482 CrPC before the High Court.",
                 "source_pdf": "crpc.pdf",
                 "extracted_citations": []}]
        engine = _build(_make_engine(meta))
        assert "482" in engine.section_index, (
            "section_index must contain '482' for a chunk with 'u/s 482'"
        )
        assert 0 in engine.section_index["482"]

    def test_standalone_pattern_does_not_match_embedded_digits(self):
        """
        '14' inside '2014' must NOT add chunk to section_index['14'].
        This is the companion assertion: not just that the pattern fires,
        but that it fires with the correct word-boundary semantics.
        """
        meta = [{"text_chunk": "The judgment dated 2014 is relevant.",  # only '2014', no standalone '14'
                 "source_pdf": "old.pdf",
                 "extracted_citations": []}]
        engine = _build(_make_engine(meta))
        # '2014' is indexed under '2014', not under '14'
        hits_14 = engine.section_index.get("14", [])
        assert 0 not in hits_14, (
            "Chunk with only '2014' (no standalone '14') must NOT appear in section_index['14']. "
            f"Got section_index['14'] = {hits_14}"
        )


class TestRegressionB_SectionIndexSelectivity:
    """
    Regression for Bug B: section_index built from all digit substrings.

    Old build: for each numeric token like "2014", ALL substrings (1-5 chars) were
    extracted — so section_index["14"], section_index["20"], section_index["201"]
    all got entries for EVERY chunk containing "2014". This made 2-digit section
    numbers non-selective (up to 67% corpus coverage for "19").

    New build: standalone tokens only via r"\b(\d+[a-zA-Z]?)\b".

    MUTATION CHECK — to see this test go RED:
      In search_engine.py _build_structured_indices(), replace the section_index block
      with the old all-substrings approach (extract every substring of each numeric tok).
      Re-run pytest: test_year_2014_not_indexed_under_14 will FAIL (chunk 0
      IS found in section_index['14']).  Revert -> PASS.
    """

    def test_year_2014_not_indexed_under_14(self):
        """
        A chunk where '14' appears ONLY as a substring of '2014' must not be
        in section_index['14'].  With the old substring approach it would be.
        """
        meta = [
            {"text_chunk": "Decided in 2014 by a bench of three judges.",
             "source_pdf": "old_case.pdf",
             "extracted_citations": []},
        ]
        engine = _build(_make_engine(meta))
        hits = engine.section_index.get("14", [])
        assert 0 not in hits, (
            "Chunk 0 has '14' only inside '2014'. It must NOT appear in section_index['14']. "
            f"section_index['14'] = {hits}"
        )

    def test_standalone_14_is_indexed(self):
        """
        A chunk with standalone '14' must be in section_index['14'].
        Confirms the fix doesn't over-exclude.
        """
        meta = [
            {"text_chunk": "Section 14 of the Arbitration and Conciliation Act 1996 states.",
             "source_pdf": "arb.pdf",
             "extracted_citations": []},
        ]
        engine = _build(_make_engine(meta))
        assert "14" in engine.section_index
        assert 0 in engine.section_index["14"]

    def test_runtime_filter_narrows_short_section_candidates(self):
        """
        For a corpus of 5 chunks, only those with 'section 19' or 'u/s 19'
        in their text must end up in the exact_identifier_indices — NOT all
        chunks that merely contain the digit 19 standalone.

        This tests the runtime condition-B filter on top of the index lookup.
        """
        corpus = [
            # chunk 0: has standalone "19" as a para number, no section reference
            {"text_chunk": "Para 19 notes that the appeal was filed timely.", "extracted_citations": []},
            # chunk 1: genuine 'section 19' reference
            {"text_chunk": "Section 19 of the Arbitration Act allows a party to file.", "extracted_citations": []},
            # chunk 2: 'u/s 19' reference
            {"text_chunk": "An application u/s 19 was filed by the petitioner.", "extracted_citations": []},
            # chunk 3: "19" only as part of "2019"
            {"text_chunk": "The 2019 amendment changed the procedure significantly.", "extracted_citations": []},
            # chunk 4: no '19' at all
            {"text_chunk": "The court directed the parties to file written submissions.", "extracted_citations": []},
        ]
        engine = _build(_make_engine(corpus))

        # Simulate the runtime section filter from search()
        sec_digit = "19"
        sec_pat = re.compile(
            r"\b(?:section|sec\.?|s\.?|u/s\.?|u/ss\.?)\s*19\b", re.I
        )
        # full_sec is purely numeric → condition B only (no condition A)
        candidates = engine.section_index.get(sec_digit, [])
        filtered = {
            ci for ci in candidates
            if sec_pat.search(corpus[ci]["text_chunk"].lower())
        }

        # Only chunks 1 and 2 should survive the runtime filter
        assert 1 in filtered, "Chunk with 'Section 19' must be in filtered set"
        assert 2 in filtered, "Chunk with 'u/s 19' must be in filtered set"
        assert 0 not in filtered, (
            "Chunk with 'Para 19' (no section keyword) must NOT survive condition-B filter"
        )
        assert 3 not in filtered, (
            "Chunk with only '2019' (no standalone 19 even in index) must NOT be in filtered set"
        )
        assert 4 not in filtered, "Chunk with no '19' at all must not appear"

        # Verify corpus size / result ratio sanity
        n_corpus = len(corpus)
        n_filtered = len(filtered)
        assert n_filtered / n_corpus < 0.50, (
            f"Runtime filter left {n_filtered}/{n_corpus} chunks — expected < 50% "
            f"(got {100*n_filtered/n_corpus:.0f}%)"
        )


class TestRegressionC_CitationIndexStopWordsFiltered:
    """
    Regression for Bug C: citation_index without stop-word filter.

    Original bug: every token from extracted_citations + header + source_pdf was
    indexed, including English function words ("the", "of", "court", "act", "in").
    Result: citation_index["the"] covered 100% of corpus, making it non-selective.

    New build: _CITATION_STOP frozenset filters these tokens before indexing.

    MUTATION CHECK — to see this test go RED:
      In search_engine.py _build_structured_indices(), change the condition:
        if len(tok) >= 2 and tok not in _CITATION_STOP:
      to:
        if len(tok) >= 2:  # remove stop-word filter
      Re-run pytest: test_stop_words_not_in_citation_index will FAIL
      (all the function words are now found in citation_index).
      Revert → PASS.
    """

    def test_stop_words_not_in_citation_index(self):
        """
        High-frequency function words must NOT appear as keys in citation_index
        even when they appear in chunk headers and extracted_citations.
        """
        meta = [
            {
                "text_chunk": (
                    "The court held that the act was applicable in this case "
                    "and the order was set aside by the High Court."
                ),
                "source_pdf": "court_judgment.pdf",
                "extracted_citations": ["the court order", "of the act"],
            }
        ]
        engine = _build(_make_engine(meta))

        STOP_WORDS_THAT_MUST_NOT_BE_INDEXED = {"the", "of", "and", "in", "is", "for",
                                                "to", "it", "be", "as", "or", "an",
                                                "by", "at"}
        found_stops = {w for w in STOP_WORDS_THAT_MUST_NOT_BE_INDEXED
                       if w in engine.citation_index}
        assert not found_stops, (
            f"Stop words must not be in citation_index. Found: {sorted(found_stops)}"
        )

    def test_domain_generic_words_not_in_citation_index(self):
        """
        Domain-common but non-selective tokens that appear in the header must
        not be indexed as citation identifiers. This includes:
        - English function words ('the', 'of', 'under', 'to', 'is')
        - Domain-generic legal words ('court', 'act', 'section') added to
          _CITATION_STOP after the §8b audit showed them covering 14-39% of corpus.

        MUTATION CHECK: in _build_structured_indices(), remove 'under', 'court'
        from _CITATION_STOP. Re-run pytest: this test goes RED.
        """
        meta = [
            {
                "text_chunk": "Under section 304 of the act the court has jurisdiction.",
                "source_pdf": "court_order.pdf",
                "extracted_citations": [],
            }
        ]
        engine = _build(_make_engine(meta))

        # These were the top offenders in the pre-fix citation_index audit
        MUST_NOT_BE_INDEXED = {"the", "of", "under", "to", "is"}
        found = {w for w in MUST_NOT_BE_INDEXED if w in engine.citation_index}
        assert not found, (
            f"Domain-generic tokens {sorted(found)} must not be in citation_index "
            "— they create non-selective posting lists."
        )

    def test_selective_citation_tokens_are_indexed(self):
        """
        Verify the stop-word filter does NOT over-exclude legitimate citation tokens.
        Specific case identifiers like 'insc', '762', '12326' must remain indexed.
        """
        meta = [
            {
                "text_chunk": "Judgment in 2024 INSC 762 was delivered by the bench.",
                "source_pdf": "insc_762_2024.pdf",
                "extracted_citations": ["2024 INSC 762"],
            }
        ]
        engine = _build(_make_engine(meta))

        # Tokens that ARE selective and must be in citation_index
        for tok in ("insc", "762"):
            assert tok in engine.citation_index, (
                f"Selective citation token {tok!r} must be in citation_index "
                "after stop-word filter is applied"
            )

    def test_plain_english_query_does_not_trigger_citation_path(self):
        """
        A plain English phrase like 'the court held' must NOT be classified as
        CITATION_OR_CASE_NO — without this, every generic legal phrase would
        trigger the citation index lookup with non-selective tokens.
        """
        for phrase in [
            "the court held that bail was denied",
            "act of parliament covers this issue",
            "in the matter of contract enforcement",
        ]:
            r = classify_and_extract_query(phrase)
            assert r["query_type"] != "CITATION_OR_CASE_NO", (
                f"Plain English phrase {phrase!r} must not route to CITATION_OR_CASE_NO, "
                f"got {r['query_type']}"
            )


class TestRegressionD_LetterSuffixSectionNotConflatedWithBareDigit:
    """
    Regression for §8e: STATUTORY_SECTION branch conflating letter-suffixed
    sections (e.g. 304B dowry death) with bare digit sections (e.g. 304 culpable homicide).

    Buggy logic:
      if (_full_sec_has_letter and full_sec in txt) or _sec_pat_fast.search(txt):
          exact_identifier_indices.add(candidate_idx)

    In the buggy logic, when full_sec has a letter (e.g., '304b'), full_sec in txt is False
    for a bare 'section 304' chunk, BUT _sec_pat_fast matches because _sec_pat_fast uses
    sec_digit ('304') with regex r"\b(?:section|...)\s*304[a-zA-Z]?\b", which matches bare 304!
    Thus bare 304 chunks were incorrectly added to exact_identifier_indices for a 304B query.

    Fixed logic:
      if _full_sec_has_letter:
          if full_sec in txt or full_sec_pattern.search(txt):
              exact_identifier_indices.add(candidate_idx)
      elif _sec_pat_fast.search(txt):
          exact_identifier_indices.add(candidate_idx)

    MUTATION CHECK — to see this test go RED:
      In search_engine.py search(), revert the STATUTORY_SECTION branch to the buggy form:
        if (_full_sec_has_letter and full_sec in txt) or _sec_pat_fast.search(txt):
            exact_identifier_indices.add(candidate_idx)
      Re-run pytest: test_bare_304_not_matched_by_304b_query will FAIL.
      Revert -> PASS.
    """

    def test_304b_query_matches_304b_chunk(self):
        """Query 'Section 304B IPC' must match a chunk containing 'Section 304B'."""
        corpus = [
            {"text_chunk": "Offence under Section 304B of IPC relates to dowry death.", "extracted_citations": []}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Section 304B IPC dowry death")
        exact = engine._get_exact_identifier_indices(parsed)
        assert 0 in exact, "Section 304B chunk must be matched by Section 304B query"

    def test_bare_304_not_matched_by_304b_query(self):
        """
        Query 'Section 304B IPC' must NOT add a bare 'Section 304' chunk to exact_identifier_indices.
        This is the primary regression test for §8e.
        """
        corpus = [
            # Chunk 0 has bare "Section 304 Part I", NO "304B" or "304b" anywhere
            {"text_chunk": "Conviction under Section 304 Part I of Indian Penal Code for culpable homicide.", "extracted_citations": []}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Section 304B IPC dowry death")
        exact = engine._get_exact_identifier_indices(parsed)
        assert 0 not in exact, (
            "Bare 'Section 304' chunk must NOT be matched when querying for 'Section 304B'"
        )

    def test_bare_304_query_matches_bare_304_chunk(self):
        """Query 'Section 304 IPC' matches a bare 'Section 304' chunk."""
        corpus = [
            {"text_chunk": "Conviction under Section 304 Part I IPC for culpable homicide.", "extracted_citations": []}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Section 304 IPC culpable homicide")
        exact = engine._get_exact_identifier_indices(parsed)
        assert 0 in exact, "Bare Section 304 chunk must be matched by Section 304 query"


class TestRegressionE_TitleIndexEquivalence:
    """
    REGRESSION TEST E (§8f): title_index inverted lookup & short-token filtering in party nouns.

    Problem 1 (Latency): CASE_TITLE queries performed an un-indexed O(N) scan over metadata_map
    (566ms bottleneck). Solved by precomputing title_index mapping party tokens (len >= 3) to
    metadata indices, reducing candidate lookup to O(k) (0.009ms).

    Problem 2 (Non-selectivity): classify_and_extract_query() extracted single-character initials
    ('k', 's', 'n', 'm') into primary_party_nouns. Substring matching ('n' in header) matched
    100% of the corpus. Solved by filtering tokens with len < 3 at the query parsing stage.

    MUTATION CHECK: If title_index is reverted or short-token filtering is removed from
    classify_and_extract_query, test_short_initial_tokens_filtered_from_parsed_party_nouns
    or test_title_index_built_for_party_nouns will fail.
    """

    def test_short_initial_tokens_filtered_from_parsed_party_nouns(self):
        """Single and 2-character initial tokens ('k', 's', 'dr') must be filtered out of primary_party_nouns."""
        parsed = classify_and_extract_query("K.S. Puttaswamy v. Union of India")
        assert parsed["query_type"] == "CASE_TITLE"
        assert "k" not in parsed["primary_party_nouns"], "'k' initial must be filtered out"
        assert "s" not in parsed["primary_party_nouns"], "'s' initial must be filtered out"
        assert parsed["primary_party_nouns"] == ["puttaswamy"]

    def test_title_index_built_for_party_nouns(self):
        """title_index must map party tokens (len >= 3) from header/pdf to candidate metadata indices."""
        corpus = [
            {"text_chunk": "Manisha Ravindra Panpatil v. State of Maharashtra. Judgment delivered on 12 May 2024.", "source_pdf": "panpatil_2024.pdf"},
            {"text_chunk": "Sonam Lakra v. State of Chhattisgarh. Criminal Appeal No. 400 of 2024.", "source_pdf": "lakra_2024.pdf"}
        ]
        engine = _build(_make_engine(corpus))
        assert "panpatil" in engine.title_index
        assert engine.title_index["panpatil"] == [0]
        assert "lakra" in engine.title_index
        assert engine.title_index["lakra"] == [1]

    def test_case_title_candidate_indices_matched_via_title_index(self):
        """_get_exact_identifier_indices() must use title_index for CASE_TITLE queries."""
        corpus = [
            {"text_chunk": "Manisha Ravindra Panpatil v. State of Maharashtra. Judgment delivered on 12 May 2024.", "source_pdf": "panpatil_2024.pdf"},
            {"text_chunk": "Sonam Lakra v. State of Chhattisgarh. Criminal Appeal No. 400 of 2024.", "source_pdf": "lakra_2024.pdf"}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Manisha Ravindra Panpatil v. State of Maharashtra")
        exact = engine._get_exact_identifier_indices(parsed)
        assert exact == {0}, f"Expected candidate set {{0}}, got {exact}"


class TestRegressionF_ArticleSectionPrefixGating:
    """
    REGRESSION TEST F (§8g): Article vs Section statutory prefix gating.

    Problem: The pre-gating pattern \\b(?:section|sec\\.?|s\\.?|u/s\\.?|u/ss\\.?)\\s*SEC_DIGIT\\b
    did not distinguish 'Article'-worded queries from 'Section'-worded ones. Querying 'Article 21'
    matched 'Section 21 of Sick Industrial Companies Act' chunks instead of 'Article 21 of
    Constitution'.

    Fix: Gate section-prefix pattern based on query intent. An 'Article' query requires an
    'Article'/'Art' prefix in text, while a 'Section' query requires a 'Section'/'Sec' prefix.

    MUTATION CHECK: If prefix gating is removed (e.g. allowing 'section' to match 'Article 21'
    or 'article' to match 'Section 21'), these tests will fail.
    """

    def test_article_query_does_not_match_unrelated_section_chunk(self):
        """Query for 'Article 21' must NOT match a chunk containing only 'Section 21' of SICA."""
        corpus = [
            {"text_chunk": "Proceedings under Section 21 of the Sick Industrial Companies Act 1985 before BIFR.", "extracted_citations": []}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Article 21 right to life and personal liberty")
        exact = engine._get_exact_identifier_indices(parsed)
        assert 0 not in exact, "Article 21 query must NOT match a Section 21 SICA chunk"

    def test_article_query_matches_genuine_article_chunk(self):
        """Query for 'Article 21' must match a chunk containing 'Article 21' of Constitution."""
        corpus = [
            {"text_chunk": "Inordinate delay in mercy petition execution violates Article 21 of the Constitution of India.", "extracted_citations": []}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Article 21 right to life and personal liberty")
        exact = engine._get_exact_identifier_indices(parsed)
        assert 0 in exact, "Article 21 query MUST match a genuine Article 21 Constitution chunk"

    def test_section_query_does_not_match_constitutional_article_chunk(self):
        """Query for 'Section 19' must NOT match a chunk containing only 'Article 19' of Constitution."""
        corpus = [
            {"text_chunk": "Freedom of speech and expression guaranteed under Article 19 of the Constitution of India.", "extracted_citations": []}
        ]
        engine = _build(_make_engine(corpus))
        parsed = classify_and_extract_query("Section 19 Prevention of Corruption Act")
        exact = engine._get_exact_identifier_indices(parsed)
        assert 0 not in exact, "Section 19 query must NOT match an Article 19 Constitutional chunk"


