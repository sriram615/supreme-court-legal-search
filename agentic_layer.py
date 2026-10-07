"""
agentic_layer.py -- LangGraph agentic layer on top of LegalSearchEngine.search().

Adds two things to the existing hybrid retrieval pipeline, without modifying
search_engine.py:

  1. Query rewriting -- but ONLY for CASE_TITLE / CONCEPTUAL_OR_AMBIGUOUS queries.
     CITATION_OR_CASE_NO and STATUTORY_SECTION queries are routed straight to
     retrieval, unmodified. search_engine.py's exact-match boosting
     (citation_index / section_index, see BUILD_REPORT.md Sec. 8e) depends on
     the identifier surviving in the query verbatim -- an LLM "helpfully"
     rewording "Section 304B IPC" into "murder amounting to culpable homicide"
     would silently break the exact-match path this project already spent
     real effort getting right. Skipping rewrite for those two intents is not
     a shortcut, it's required for correctness, and it's why this graph calls
     classify_and_extract_query() itself before deciding whether to rewrite.

  2. Answer synthesis -- an LLM reads the top-k retrieved chunks (with their
     chunk_id / source_pdf metadata) and produces a short, cited answer, or
     explicitly says the corpus doesn't contain enough information rather
     than hallucinating one.

Both LLM steps degrade gracefully: if no LLM credentials are configured, or
the LLM call fails for any reason (bad key, timeout, rate limit), this falls
back to the engine's raw hits with a None answer instead of raising. The
existing /api/v1/analyze endpoint must keep working even when this layer
can't run.

STATUS: written directly against search_engine.py's real interface (search(),
classify_and_extract_query()), read from source in this repo -- but NOT yet
run against the real FAISS index / corpus, since this environment has
neither. The routing logic (which queries skip rewriting) is unit-tested
without needing the real index or an API key -- see test_agentic_layer.py.
The end-to-end LLM behavior (rewrite quality, synthesis quality, whether it
actually improves answers) is NOT verified yet. Do not claim it works well
until you've run it against real queries and read the actual output --
same discipline the rest of this repo already follows.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM client. If validator.py already authenticates against a provider, reuse
# that one instead of adding a second API key for one feature -- set
# LLM_PROVIDER=openai or LLM_PROVIDER=anthropic to match it.
#
# Default here is "groq": a genuinely free, no-credit-card tier (as of writing
# -- verify at console.groq.com, terms change) that's enough to develop and
# demo this feature without paying for anything. 30 req/min, 1,000 req/day on
# Llama 3.3 70B, no training on your prompts on the free tier. Good enough for
# both the rewrite step (easy task) and synthesis (needs decent instruction-
# following, which 70B-class Llama handles fine for 2-5 sentence cited
# answers). Get a key at console.groq.com -> API Keys, no card required, set
# GROQ_API_KEY in your environment.
#
# Alternative free option: Google AI Studio / Gemini (LLM_PROVIDER=google) --
# bigger context window and arguably higher quality, also no card required,
# but Google trains on your prompts on the free tier unless you're in the
# EU/UK/EEA. Fine here since the excerpts are already-public court judgment
# text, not private data, but worth knowing before you pick it for something
# else.
# ─────────────────────────────────────────────────────────────────────────────
def _get_chat_model():
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.environ.get("LLM_MODEL", "claude-3-5-haiku-latest"), temperature=0)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.environ.get("LLM_MODEL", "gemini-2.0-flash"), temperature=0)
    # default / provider == "groq". llama-3.3-70b-versatile now 404s on standard Groq
    # keys (see validator.py), so default to the same model deploy/gcp/deploy.sh sets.
    from langchain_groq import ChatGroq
    return ChatGroq(model=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"), temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    query_type: str
    parsed: Dict[str, Any]
    rewritten_query: Optional[str]
    hits: List[Dict[str, Any]]
    answer: Optional[str]
    used_chunk_ids: List[str]
    error: Optional[str]


REWRITE_PROMPT = """You are a query normalizer for a hybrid BM25 + vector search engine over Indian Supreme Court judgments.
Rewrite the user's question into a short, retrieval-friendly query: expand abbreviations, add the specific legal
concept/doctrine name if implied, and drop conversational filler. Do NOT invent citations, section numbers, or case
names that are not already implied by the question. Return ONLY the rewritten query text, nothing else.

User question: {query}"""

SYNTHESIS_PROMPT = """You are a legal research assistant. Answer the user's question using ONLY the excerpts below,
each tagged with a chunk id. Cite the chunk id(s) you relied on in square brackets, e.g. [chunk_id: abc123].
If the excerpts do not contain enough information to answer, say so explicitly instead of guessing.

Question: {query}

Excerpts:
{excerpts}

Answer (2-5 sentences, with inline [chunk_id: ...] citations):"""


def _classify_node(state: AgentState) -> AgentState:
    from search_engine import classify_and_extract_query  # local import: no hard dep at module load
    parsed = classify_and_extract_query(state["query"])
    return {**state, "parsed": parsed, "query_type": parsed.get("query_type", "CONCEPTUAL_OR_AMBIGUOUS")}


def _route_after_classify(state: AgentState) -> str:
    # Exact-match intents must not be reworded -- see module docstring.
    if state["query_type"] in ("CITATION_OR_CASE_NO", "STATUTORY_SECTION"):
        return "retrieve"
    return "rewrite"


def _rewrite_node(state: AgentState) -> AgentState:
    try:
        llm = _get_chat_model()
        resp = llm.invoke(REWRITE_PROMPT.format(query=state["query"]))
        rewritten = str(resp.content).strip()
        if not rewritten:
            rewritten = state["query"]
    except Exception as exc:  # LLM down, bad key, network -- never break retrieval
        logger.warning("Query rewrite failed, falling back to original query: %s", exc)
        rewritten = state["query"]
    return {**state, "rewritten_query": rewritten}


def _retrieve_node(state: AgentState, engine) -> AgentState:
    query_for_search = state.get("rewritten_query") or state["query"]
    hits = engine.search(query_for_search, top_k=state.get("top_k", 5), mode="hybrid")
    return {**state, "hits": hits}


def _synthesize_node(state: AgentState) -> AgentState:
    hits = state.get("hits") or []
    if not hits:
        return {**state, "answer": None, "used_chunk_ids": []}
    excerpts = "\n\n".join(
        f"[chunk_id: {h.get('chunk_id', '?')}] ({h.get('source_pdf', '?')})\n{h.get('text_chunk', '')[:800]}"
        for h in hits
    )
    try:
        llm = _get_chat_model()
        resp = llm.invoke(SYNTHESIS_PROMPT.format(query=state["query"], excerpts=excerpts))
        answer = str(resp.content).strip()
        used = list(dict.fromkeys(re.findall(r"chunk_id:\s*([^\]]+)\]", answer)))
        return {**state, "answer": answer, "used_chunk_ids": used}
    except Exception as exc:
        logger.warning("Answer synthesis failed, returning hits without a synthesized answer: %s", exc)
        return {**state, "answer": None, "used_chunk_ids": [], "error": str(exc)}


def build_agentic_graph(engine):
    """
    engine: an already-loaded LegalSearchEngine instance
    (engine.load_index() must already have been called -- same precondition
    as calling engine.search() directly).
    """
    graph = StateGraph(AgentState)
    graph.add_node("classify", _classify_node)
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("retrieve", lambda s: _retrieve_node(s, engine))
    graph.add_node("synthesize", _synthesize_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _route_after_classify, {"rewrite": "rewrite", "retrieve": "retrieve"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_agentic_search(engine, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Drop-in enrichment of engine.search(): same hits, plus (when an LLM is
    configured and reachable) a rewritten_query and a synthesized, cited
    answer. Never raises -- on any internal failure, falls back to plain
    engine.search() so the caller always gets a usable result.
    """
    try:
        compiled = build_agentic_graph(engine)
        result = compiled.invoke({"query": query, "top_k": top_k})
        return {
            "query_type": result.get("query_type"),
            "rewritten_query": result.get("rewritten_query"),
            "hits": result.get("hits", []),
            "answer": result.get("answer"),
            "used_chunk_ids": result.get("used_chunk_ids", []),
        }
    except Exception as exc:
        logger.error("Agentic layer failed end-to-end, falling back to plain retrieval: %s", exc)
        return {
            "query_type": None,
            "rewritten_query": None,
            "hits": engine.search(query, top_k=top_k, mode="hybrid"),
            "answer": None,
            "used_chunk_ids": [],
        }


if __name__ == "__main__":
    # Manual smoke test -- run with: python agentic_layer.py
    # Requires the real venv (search_engine's deps) and a built index.
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    from search_engine import LegalSearchEngine

    engine = LegalSearchEngine(index_dir="./LAWdata_Corpus_2024")
    engine.load_index()

    for test_query in [
        "2024 INSC 762",                              # CITATION_OR_CASE_NO -- must NOT be rewritten
        "Section 304B IPC essential ingredients",      # STATUTORY_SECTION -- must NOT be rewritten
        "what makes a dowry death case",               # CONCEPTUAL -- should be rewritten
    ]:
        print("=" * 80)
        print("QUERY:", test_query)
        out = run_agentic_search(engine, test_query, top_k=5)
        print("query_type:      ", out["query_type"])
        print("rewritten_query: ", out["rewritten_query"])
        print("num hits:        ", len(out["hits"]))
        print("answer:          ", out["answer"])
        print("used_chunk_ids:  ", out["used_chunk_ids"])
