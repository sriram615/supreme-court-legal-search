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
  against the top matching precedent and flags disagreement.
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
| `/api/v1/analyze` | POST | Full pipeline: retrieval + citation resolution + contradiction verification |
| `/api/v1/health` | GET | Engine readiness, vector count, citation-map size |

## Testing & evaluation

- 92-case automated regression suite (`pytest tests/`), including mutation-tested cases that
  verify each regression test actually fails against the bug it targets, not just passes
  against the fix.
- An independent 120-query retrieval-quality evaluation (30 queries per intent type),
  measuring Hit@5, MRR@5, and Precision@5 in hybrid mode — current results: **76.7% Hit@5**,
  **0.649 MRR@5** overall.

## Project structure

```
search_engine.py       Core retrieval engine: indices, classification, hybrid search
app.py                 FastAPI service (analyze / health endpoints)
frontend/               Web console (static HTML/CSS/JS, served by app.py)
tests/                 Pytest regression suite
data/, eval/           Evaluation query set and retrieval-quality results
validator.py           LLM-based precedent-contradiction verification
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
Pytest.
