"""
test_agentic_layer.py -- unit tests for agentic_layer.py's routing logic.

These tests need NO real LLM API key and NO real FAISS index: they mock both
the chat model and the search engine, and check the one thing that actually
matters for correctness -- that CITATION_OR_CASE_NO / STATUTORY_SECTION
queries never get sent through the rewrite node, so the engine's exact-match
indices always see the identifier verbatim. Mirrors this repo's existing
test discipline (see tests/test_search_engine.py): test the routing decision
itself, not just that *a* response comes back.

Run with: HF_HUB_OFFLINE=1 .venv311/bin/python -m pytest tests/test_agentic_layer.py -v
(HF_HUB_OFFLINE isn't actually needed here since these tests never touch the
real model/index, but kept for consistency with how the rest of the suite is
invoked in this repo.)
"""
from unittest.mock import MagicMock

import pytest

from agentic_layer import run_agentic_search


class _FakeEngine:
    """Stands in for LegalSearchEngine: records what query text it was called
    with, so tests can assert on it without a real index."""

    def __init__(self, hits):
        self._hits = hits
        self.search_calls = []

    def search(self, query, top_k=5, mode="hybrid"):
        self.search_calls.append(query)
        return self._hits


@pytest.fixture
def fake_hits():
    return [
        {
            "chunk_id": "c1",
            "source_pdf": "case1.pdf",
            "text_chunk": "Section 304B deals with dowry death and requires death within seven years of marriage...",
        }
    ]


def test_citation_query_skips_rewrite(fake_hits, monkeypatch):
    """A citation query's exact text must reach engine.search() unmodified --
    citation_index lookups depend on it."""
    engine = _FakeEngine(fake_hits)
    monkeypatch.setattr("agentic_layer._get_chat_model", lambda: MagicMock())

    result = run_agentic_search(engine, "2024 INSC 762", top_k=3)

    assert result["rewritten_query"] is None
    assert engine.search_calls == ["2024 INSC 762"]


def test_statutory_section_query_skips_rewrite(fake_hits, monkeypatch):
    """Same guarantee for a statutory-section query -- section_index lookups
    depend on the exact digit/letter-suffix string surviving untouched."""
    engine = _FakeEngine(fake_hits)
    monkeypatch.setattr("agentic_layer._get_chat_model", lambda: MagicMock())

    result = run_agentic_search(engine, "Section 304B IPC essential ingredients", top_k=3)

    assert result["rewritten_query"] is None
    assert engine.search_calls == ["Section 304B IPC essential ingredients"]


def test_conceptual_query_is_routed_through_rewrite(fake_hits, monkeypatch):
    """A conceptual query has no exact-match index to protect, so it's fine
    for it to be rewritten -- and the rewritten text (not the original)
    should be what reaches engine.search()."""
    engine = _FakeEngine(fake_hits)

    class _FakeLLM:
        def invoke(self, prompt):
            msg = MagicMock()
            msg.content = "dowry death essential ingredients Section 304B"
            return msg

    monkeypatch.setattr("agentic_layer._get_chat_model", lambda: _FakeLLM())

    result = run_agentic_search(engine, "what counts as dowry death", top_k=3)

    assert result["rewritten_query"] == "dowry death essential ingredients Section 304B"
    assert engine.search_calls == ["dowry death essential ingredients Section 304B"]


def test_llm_failure_falls_back_to_plain_retrieval(fake_hits, monkeypatch):
    """If the LLM is unreachable (no API key, network error, rate limit),
    the endpoint must still return search results -- never raise."""
    engine = _FakeEngine(fake_hits)

    def _broken_llm():
        raise RuntimeError("no API key configured")

    monkeypatch.setattr("agentic_layer._get_chat_model", _broken_llm)

    result = run_agentic_search(engine, "what counts as dowry death", top_k=3)

    assert result["hits"] == fake_hits
    assert result["answer"] is None


def test_empty_hits_short_circuits_synthesis(monkeypatch):
    """No point calling an LLM to synthesize an answer from zero chunks."""
    engine = _FakeEngine([])
    llm = MagicMock()
    monkeypatch.setattr("agentic_layer._get_chat_model", lambda: llm)

    result = run_agentic_search(engine, "an extremely obscure query", top_k=3)

    assert result["hits"] == []
    assert result["answer"] is None
    assert result["used_chunk_ids"] == []
