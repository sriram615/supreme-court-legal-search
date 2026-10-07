---
title: Indian Supreme Court Legal Search Engine
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
---

# Supreme Court Legal Search Engine

A hybrid semantic + lexical search engine over ~15,800 chunks of real Indian Supreme Court
judgment text, exposed as a FastAPI service with an interactive web console. Built to
correctly route four distinct query intents — case citations, case titles, statutory
sections, and open-ended conceptual questions — each through retrieval logic suited to it,
rather than treating every query as a plain semantic-similarity lookup.

## What it does

- **Search** — case titles, statutory sections (e.g. "Section 304B IPC"), citations (e.g.
  "2024 INSC 762" or "(2024) 9 SCC 1"), or a free-text legal question.
- **Citation resolution** — extracts and resolves citation strings against a canonical
  citation map.
- **Precedent-contradiction check** — runs a referee LLM pass comparing a submitted brief
  against the top matching precedent and flags a possible contradiction. The referee tries a
  local Ollama model (`phi3`) first, then falls back to Groq if `GROQ_API_KEY` is set. If
  neither is reachable the verdict is returned with `available: false` and the console shows
  "Verification unavailable" instead of a result.
- **Agentic search** — an optional LangGraph pipeline that rewrites conceptual / case-title
  queries and synthesizes a short answer citing chunk ids. Citation and statutory-section
  queries are never rewritten, so exact-match retrieval always sees the identifier verbatim.
  If no LLM is configured it falls back to plain retrieval (`synthesized_answer: null`).
- **Web console** — a chat-style interface for interactive search, with query history and
  live engine status.

## Architecture

**Query classification.** Every query is classified into one of four intents
(`CITATION_OR_CASE_NO`, `CASE_TITLE`, `STATUTORY_SECTION`, `CONCEPTUAL_OR_AMBIGUOUS`) before
retrieval runs, since each needs different handling — a citation has one right answer to
find exactly; a conceptual question doesn't.

**Structured inverted indices.** Three indices are built once at startup and reused across
queries:
- `citation_index` — tokens from citations, source filenames, and document headers, with
  domain-specific stop-word filtering (English function words plus generic legal terms like
  "court," "section," "act") to keep posting lists selective.
- `section_index` — standalone statutory-section digit tokens (e.g. "304", "482"), matched
  with a word-boundary-aware pattern so "14" inside "2014" is never indexed as section 14.
- `title_index` — case-party-name tokens (length ≥ 3, filtering out single-letter initials)
  mapped to candidate chunks.

These turn what would be an O(N) scan of the full corpus per query into an O(k) lookup
against a small candidate set, which then gets verified and boosted rather than trusted
blindly — e.g. a `STATUTORY_SECTION` candidate is only boosted if the *exact* section
(including any letter suffix, so "304B" is never conflated with bare "304") is actually
present, and an "Article" query is only matched against "Article" text, never "Section" text
sharing the same digit.

**Hybrid ranking.** BM25 (lexical) and FAISS `IndexHNSWFlat` (dense vector, cosine similarity
over 384-dim `all-MiniLM-L6-v2` embeddings) are run independently, boosted with the exact-
match candidates above, and fused via Reciprocal Rank Fusion — a chunk ranking well in either
signal contributes to the final rank, rather than requiring agreement from both.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/analyze` | POST | Full pipeline: retrieval (top 3) + citation resolution + contradiction verification |
| `/api/v1/agentic-analyze` | POST | Intent-aware query rewriting + cited answer synthesis (needs an LLM key; degrades to plain retrieval) |
| `/api/v1/health` | GET | Engine readiness, vector count, citation-map size |

The referee verdict has the shape `{verdict_agreement, legal_rationale, confidence_rating, available}`;
`verdict_agreement: true` means a contradiction was found, and `available: false` means no referee
produced the result. Error responses are generic (details are written to the server log only).
The web console currently uses `/api/v1/analyze` only.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Enables the Groq referee fallback and the default agentic LLM |
| `GROQ_REFEREE_MODEL` | `openai/gpt-oss-20b` | Groq model for the referee |
| `LLM_PROVIDER` | `groq` | Agentic layer provider: `groq`, `openai`, `anthropic`, `google` |
| `LLM_MODEL` | `openai/gpt-oss-20b` (groq) | Agentic layer model |

Note: when the Groq fallback is used, the submitted brief text is sent to Groq's API.

## Testing & evaluation

- 97-case automated regression suite (`pytest tests/`), including mutation-tested cases that
  verify each regression test actually fails against the bug it targets, not just passes
  against the fix. The agentic-layer tests cover routing only (no real LLM or index).
- An independent 120-query retrieval-quality evaluation (30 queries per intent type),
  measuring Hit@5, MRR@5, and Precision@5 in hybrid mode — current results (RRF exact-match
  weight 2.5): **76.7% Hit@5**, **0.649 MRR@5** overall. These were measured on the same set
  used while tuning, so treat them as development-set numbers, not held-out performance.
  `eval/retrieval_eval_report.md` is an earlier snapshot (5.0 weight, 75.0% / 0.664) kept for
  reference; `BUILD_REPORT.md` §8i has the authoritative comparison.
- `eval/precedent_referee_eval.py` scores the referee on 20 hand-labeled argument/precedent
  pairs (a small sample).

## Project structure

```
search_engine.py       Core retrieval engine: indices, classification, hybrid search
app.py                 FastAPI service (analyze / agentic-analyze / health endpoints)
agentic_layer.py       LangGraph query rewriting + cited answer synthesis
frontend/               Web console (static HTML/CSS/JS, served by app.py)
tests/                 Pytest regression suite
data/, eval/           Evaluation query set and retrieval-quality results
validator.py           LLM-based precedent-contradiction verification (Ollama, Groq fallback)
fetch_and_preprocess.py  Corpus ingestion / chunking pipeline
BUILD_REPORT.md        Detailed build, benchmark, and verification log
```

## Running it

```bash
pip install -r requirements.txt
python search_engine.py          # builds the FAISS index + embeddings (first run only)
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the web console, or call `/api/v1/analyze` directly.

## Tech stack

Python (Asyncio), FastAPI, Pydantic v2, PyTorch, Hugging Face Transformers, FAISS, BM25,
LangGraph, Pytest. Dependencies are pinned in `requirements.txt`.
