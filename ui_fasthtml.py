"""
ui_fasthtml.py
=============
Ultra-modern FastHTML dashboard interface echoing the premium design language
of Anthropic Claude and Shadcn UI.

Runs on port 8001. Connects to backend FastAPI server at http://127.0.0.1:8000.
"""

import httpx
import logging
from fasthtml.common import *

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ─ CONFIGURATION & INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SC_UI")

# Custom headers for Tailwind CSS, Lucide Icons, and HTMX customization
headers = (
    Meta(charset="UTF-8"),
    Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
    Title("Contradiction Radar v1.0 — Legal Analytics Dashboard"),
    # Core Tailwind CSS
    Script(src="https://cdn.tailwindcss.com"),
    # Lucide Icon library
    Script(src="https://unpkg.com/lucide@latest"),
    # Inter Font family for premium typographic scale
    Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap", rel="stylesheet"),
    Style("""
        body {
            font-family: 'Inter', sans-serif;
            background-color: #020617; /* bg-slate-950 */
        }
        .htmx-indicator {
            display: none;
        }
        .htmx-request .htmx-indicator {
            display: inline-block;
        }
        .htmx-request.htmx-indicator {
            display: block;
        }
    """),
    # Trigger Lucide icons rendering on dynamic content load
    Script("""
        document.addEventListener('DOMContentLoaded', function() {
            lucide.createIcons();
        });
        document.body.addEventListener('htmx:afterSwap', function(evt) {
            lucide.createIcons();
        });
    """)
)

# Initialize FastHTML with Pico CSS disabled to ensure vanilla Tailwind takes complete control
app, rt = fast_app(pico=False, hdrs=headers)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ TEMPLATES AND UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def layout_wrapper(sidebar_content, main_content):
    """Generates the master dual-column split workspace."""
    return Div(
        # Root container
        Div(
            # Sidebar Left (20% width)
            Div(
                sidebar_content,
                cls="w-1/5 bg-slate-900 border-r border-slate-800 p-5 flex flex-col justify-between h-screen sticky top-0"
            ),
            # Main Window Area (80% width)
            Div(
                main_content,
                cls="w-4/5 p-8 overflow-y-auto h-screen text-slate-200"
            ),
            cls="flex min-h-screen"
        ),
        cls="bg-slate-950 text-slate-200"
    )

def render_sidebar():
    """Generates the static control and dynamic health panel."""
    return Div(
        Div(
            # App Branding
            Div(
                Span("CONTRADICTION", cls="text-slate-100 font-bold tracking-wider text-sm"),
                Span("RADAR v1.0", cls="text-amber-400 font-bold tracking-wider text-xs ml-1.5"),
                cls="flex items-center pb-6 border-b border-slate-800 mb-6"
            ),
            # Navigation / Menu placeholder
            Div(
                Div(
                    A(
                        Span(I(data_lucide="terminal", cls="w-4 h-4 mr-2.5 text-amber-500")),
                        Span("Judicial Analyzer"),
                        href="#",
                        cls="flex items-center text-xs font-semibold uppercase tracking-wider text-amber-400 bg-amber-950/20 border border-amber-900/30 px-3 py-2.5 rounded-md mb-2"
                    ),
                    cls="space-y-1"
                ),
                cls="mb-8"
            ),
            # Asynchronous Health Panel
            Div(
                Div(
                    Div(
                        I(data_lucide="loader-2", cls="w-4 h-4 mr-2 text-slate-400 animate-spin htmx-indicator"),
                        Span("Querying engine health..."),
                        cls="flex items-center text-xs text-slate-400 p-3 bg-slate-950/40 border border-slate-850 rounded-lg"
                    ),
                    hx_get="/local-health",
                    hx_trigger="load",
                    hx_swap="innerHTML",
                    cls="mt-4"
                ),
                cls="space-y-3"
            ),
        ),
        # Sidebar Footer
        Div(
            Div(
                P("Supreme Court of India", cls="text-xs font-semibold text-slate-400"),
                P("Local Evaluation Instance", cls="text-[10px] text-slate-500 mt-0.5"),
                cls="pt-4 border-t border-slate-800"
            ),
        ),
        cls="flex flex-col justify-between h-full"
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ ROUTING AND INTERACTION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@rt("/")
async def get():
    sidebar = render_sidebar()
    
    # Default: Tab 1 (Workspace Analysis)
    main = Div(
        # Tab selection switcher
        Div(
            Button(
                "Workspace Analysis",
                hx_get="/tab/workspace",
                hx_target="#tab-content-container",
                hx_swap="innerHTML",
                cls="px-5 py-2.5 text-xs font-semibold tracking-wider uppercase text-amber-400 border-b-2 border-amber-500 outline-none transition-colors"
            ),
            Button(
                "Precedent Explorer Matrix",
                hx_get="/tab/explorer",
                hx_target="#tab-content-container",
                hx_swap="innerHTML",
                cls="px-5 py-2.5 text-xs font-semibold tracking-wider uppercase text-slate-400 hover:text-slate-200 border-b-2 border-transparent outline-none transition-colors ml-4"
            ),
            cls="flex border-b border-slate-800 mb-8"
        ),
        # Core Content Area (loads Tab 1 by default)
        Div(
            render_workspace_analysis_tab(),
            id="tab-content-container"
        )
    )
    return layout_wrapper(sidebar, main)

@rt("/tab/workspace")
async def get():
    """Route handler for Workspace Analysis Tab."""
    return render_workspace_analysis_tab()

@rt("/tab/explorer")
async def get():
    """Route handler for Precedent Explorer Tab."""
    return render_precedent_explorer_tab()

def render_workspace_analysis_tab():
    """Renders the document submission container."""
    return Div(
        H2("Judicial Decision Verification Workspace", cls="text-lg font-bold text-slate-100 tracking-tight mb-2"),
        P("Submit a raw case brief snippet, defense pleading, or legal argument to execute deep precedent search and verify logical contradictions against Year 2024 Supreme Court holdings.", cls="text-xs text-slate-400 mb-6"),
        
        Form(
            Div(
                Textarea(
                    name="text",
                    placeholder="Enter brief text snippet or legal argument to analyze...",
                    cls="w-full h-80 bg-slate-900 border border-slate-800 rounded-lg p-4 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-amber-500/50 focus:border-amber-500 resize-none font-mono tracking-tight transition-all"
                ),
                cls="relative border border-slate-800 rounded-lg overflow-hidden bg-slate-900"
            ),
            Div(
                # Loading Spinner Indicator
                Div(
                    I(data_lucide="loader-2", cls="w-4 h-4 mr-2 animate-spin text-amber-500"),
                    Span("Analyzing legal context and retrieval vectors...", cls="text-xs text-slate-400 font-medium"),
                    id="loading-spinner",
                    cls="htmx-indicator flex items-center bg-slate-900/50 px-3 py-1.5 border border-slate-800 rounded-md"
                ),
                # Submit Action Button
                Button(
                    Span("Analyze Legal Brief"),
                    I(data_lucide="sparkles", cls="w-3.5 h-3.5 ml-2"),
                    type="submit",
                    cls="flex items-center text-xs font-semibold tracking-wider uppercase bg-amber-600 hover:bg-amber-500 text-slate-950 px-5 py-3 rounded-md transition-all shadow-lg hover:shadow-amber-900/20 outline-none ml-auto"
                ),
                cls="flex items-center justify-between mt-4"
            ),
            hx_post="/analyze",
            hx_target="#analysis-response-container",
            hx_swap="innerHTML",
            hx_indicator="#loading-spinner"
        ),
        # Response Container
        Div(
            id="analysis-response-container",
            cls="mt-8 border-t border-slate-800/80 pt-6"
        )
    )

def render_precedent_explorer_tab():
    """Renders a search-and-browse matrix over indexed vector metadata."""
    return Div(
        H2("Precedent Explorer Matrix", cls="text-lg font-bold text-slate-100 tracking-tight mb-2"),
        P("Run direct semantic searches against the HNSW FAISS vector database to explore the case law database directly.", cls="text-xs text-slate-400 mb-6"),
        
        Form(
            Div(
                Input(
                    type="text",
                    name="query",
                    placeholder="Search keywords, case issues, or judge names (e.g. encroachment, female Sarpanch)...",
                    cls="w-full bg-slate-900 border border-slate-800 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-amber-500/50 focus:border-amber-500 transition-all"
                ),
                Button(
                    I(data_lucide="search", cls="w-4 h-4 text-slate-950"),
                    type="submit",
                    cls="absolute right-2 top-2 bg-amber-500 hover:bg-amber-400 p-1.5 rounded-md transition-all focus:outline-none"
                ),
                cls="relative w-full mb-6"
            ),
            hx_post="/explorer-search",
            hx_target="#explorer-results-container",
            hx_swap="innerHTML"
        ),
        # Results Output Grid
        Div(
            Div(
                I(data_lucide="compass", cls="w-8 h-8 text-slate-700 mb-2"),
                P("Submit a semantic query above to inspect similarity rankings in the database.", cls="text-xs text-slate-500"),
                cls="flex flex-col items-center justify-center border border-dashed border-slate-800 py-16 rounded-lg bg-slate-900/10"
            ),
            id="explorer-results-container"
        )
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ SYSTEM PROXIES (FASTAPI BRIDGE)
# ─────────────────────────────────────────────────────────────────────────────

@rt("/local-health")
async def get():
    """Proxies GET requests to FastAPI /health check and formats response card."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://127.0.0.1:8000/api/v1/health", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            is_ready = data.get("engine_ready", False)
            vectors = data.get("index_vectors_count", 0)
            threads = data.get("thread_tuning", {}).get("torch_threads", 4)
            
            status_color = "text-emerald-400" if is_ready else "text-amber-400"
            status_bg = "bg-emerald-500/10 border-emerald-500/30" if is_ready else "bg-amber-500/10 border-amber-500/30"
            dot_color = "bg-emerald-400" if is_ready else "bg-amber-400"
            
            return Div(
                Div(
                    Span(cls=f"w-2 h-2 rounded-full {dot_color} animate-ping mr-2"),
                    Span("Engine Active" if is_ready else "Degraded", cls=f"text-xs font-bold tracking-wider uppercase {status_color}"),
                    cls=f"flex items-center p-3 rounded-lg border {status_bg} mb-4"
                ),
                Div(
                    Div(
                        Div("Vector Store Index", cls="text-[10px] text-slate-500 uppercase tracking-wider"),
                        Div(f"{vectors:,} chunks", cls="text-base font-semibold text-slate-200 mt-0.5"),
                        cls="pb-2.5 border-b border-slate-800"
                    ),
                    Div(
                        Div("Torch Math Threads", cls="text-[10px] text-slate-500 uppercase tracking-wider mt-2.5"),
                        Div(f"{threads} active threads", cls="text-xs font-semibold text-slate-300 mt-0.5"),
                        cls="pb-2.5 border-b border-slate-800"
                    ),
                    Div(
                        Div("Processor Type", cls="text-[10px] text-slate-500 uppercase tracking-wider mt-2.5"),
                        Div("Intel CPU Optimized", cls="text-xs font-semibold text-slate-300 mt-0.5"),
                    ),
                    cls="space-y-1 p-4 bg-slate-950/60 border border-slate-850 rounded-lg"
                )
            )
        else:
            return Div("Engine Degraded (Http 503)", cls="text-xs text-red-400 p-3 bg-red-950/20 border border-red-900/30 rounded-lg")
    except Exception as exc:
        logger.error("Health proxy failed: %s", exc)
        return Div(
            Div(
                Span(cls="w-2 h-2 rounded-full bg-red-500 mr-2"),
                Span("Engine Offline", cls="text-xs font-bold tracking-wider uppercase text-red-400"),
                cls="flex items-center p-3 rounded-lg border border-red-500/30 bg-red-500/10 mb-2"
            ),
            P("Start uvicorn on port 8000.", cls="text-[10px] text-slate-500 text-center"),
            cls="flex flex-col"
        )

@rt("/analyze")
async def post(text: str):
    """Proxies POST submissions to FastAPI and formats decision report."""
    if not text.strip():
        return Div("Please input non-empty legal text snippet for validation.", cls="text-xs text-slate-400 p-4 border border-slate-800 rounded bg-slate-900/20")
    
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post("http://127.0.0.1:8000/api/v1/analyze", json={"text": text}, timeout=15.0)
        
        if r.status_code != 200:
            return Div(f"Analysis Engine Error (HTTP {r.status_code})", cls="text-sm text-red-400 p-4 bg-red-950/20 border border-red-900/30 rounded-lg")
        
        res = r.json()
        contradiction_prob = res.get("contradiction_probability", 0.0)
        is_contradiction = res.get("is_contradiction", False)
        citations = res.get("resolved_citations", {})
        matches = res.get("semantic_matches", [])
        verdict = res.get("verification_verdict")

        # ── 1. LLM Verification Verdict Card ──
        verdict_card = ""
        if verdict:
            verdict_agreement = verdict.get("verdict_agreement", False)
            rationale = verdict.get("legal_rationale", "")
            confidence = verdict.get("confidence_rating", 0.0)
            
            if verdict.get("available") is False or "unavailable" in rationale.lower():
                verdict_card = Div(
                    Div(
                        Div(
                            I(data_lucide="info", cls="w-4 h-4 mr-2 text-slate-400 inline"),
                            Span("LLM Verification Status:", cls="text-slate-300 font-semibold text-xs"),
                            Span("UNAVAILABLE", cls="text-[10px] font-bold px-2 py-0.5 rounded border ml-2.5 bg-slate-800 border-slate-700 text-slate-400"),
                            cls="flex items-center"
                        ),
                        P(rationale, cls="text-xs text-slate-400 mt-2 leading-relaxed font-mono"),
                        cls="p-4 bg-slate-900/40 border border-slate-800 rounded-lg mb-6"
                    )
                )
            else:
                verdict_status = "CONFIRMED" if verdict_agreement else "DISMISSED"
                verdict_color = "text-red-400" if verdict_agreement else "text-emerald-400"
                badge_bg = "bg-red-500/10 border-red-500/30" if verdict_agreement else "bg-emerald-500/10 border-emerald-500/30"
                
                verdict_card = Div(
                    Div(
                        Div(
                            Div(
                                I(data_lucide="gavel", cls="w-4 h-4 mr-2 text-amber-500 inline"),
                                Span("LLM Governance Referee Verdict:", cls="text-slate-200 font-semibold text-xs"),
                                Span(verdict_status, cls=f"text-[10px] font-bold px-2 py-0.5 rounded border ml-2.5 {badge_bg} {verdict_color}"),
                                cls="flex items-center"
                            ),
                            Span(f"Confidence: {confidence*100:.1f}%", cls="text-[10px] text-slate-400 font-mono"),
                            cls="flex items-center justify-between border-b border-slate-800 pb-2 mb-3"
                        ),
                        P(rationale, cls="text-xs text-slate-350 leading-relaxed font-mono"),
                        cls="p-4 bg-slate-900/40 border border-slate-850 rounded-lg mb-6"
                    )
                )

        # ── 2. Citation Chips ──
        chips_list = []
        if citations:
            for raw_cit, token_res in citations.items():
                chips_list.append(
                    Div(
                        I(data_lucide="file-text", cls="w-3.5 h-3.5 text-amber-500 mr-1.5 inline"),
                        Span(raw_cit, cls="text-slate-200 font-semibold mr-1"),
                        Span(f"→ {token_res.split(' -> ')[1] if ' -> ' in token_res else token_res}", cls="text-slate-400 text-[11px]"),
                        cls="flex items-center bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-slate-300 font-mono tracking-tight"
                    )
                )
        
        citations_section = Div(
            H3("Extracted Citation Footprints", cls="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3"),
            Div(
                *chips_list,
                cls="flex flex-wrap gap-2"
            ) if chips_list else P("No canonical citations detected in brief snippet.", cls="text-xs text-slate-500 italic"),
            cls="mb-6 pb-6 border-b border-slate-900"
        )

        # ── 3. Semantic Reference Matches (Collapsible Cards) ──
        match_components = []
        for match in matches:
            score = match.get("cosine_similarity", 0.0)
            court = match.get("court", "Supreme Court of India")
            pdf = match.get("source_pdf", "Unknown")
            citations_list = ", ".join(match.get("extracted_citations", [])) or "None"
            chunk_id = match.get("chunk_id", "N/A")
            text_snippet = match.get("text_chunk", "").strip()

            match_components.append(
                Details(
                    Summary(
                        Div(
                            Div(
                                Span(court, cls="text-xs font-semibold text-slate-200"),
                                Span(f"Case Ref: {pdf.replace('_EN.pdf', '')}", cls="text-[10px] text-slate-500 font-mono ml-2.5"),
                                cls="flex items-center"
                            ),
                            Div(
                                Span(f"Relevance: {score*100:.1f}%", cls="text-xs font-bold text-amber-400 font-mono"),
                                I(data_lucide="chevron-down", cls="w-3.5 h-3.5 text-slate-500 ml-2.5 inline"),
                                cls="flex items-center"
                            ),
                            cls="flex items-center justify-between cursor-pointer py-3.5 px-4 outline-none select-none bg-slate-900/70"
                        )
                    ),
                    Div(
                        Div(
                            Div(
                                Div(
                                    Span("Chunk ID:", cls="text-[10px] text-slate-500 uppercase"),
                                    Span(chunk_id, cls="text-[10px] text-slate-350 font-mono ml-1.5"),
                                    cls="mr-6"
                                ),
                                Div(
                                    Span("Citations Mapped:", cls="text-[10px] text-slate-500 uppercase"),
                                    Span(citations_list, cls="text-[10px] text-amber-400/80 font-mono ml-1.5"),
                                ),
                                cls="flex items-center mb-3 text-[10px] border-b border-slate-800 pb-2"
                            ),
                            P(text_snippet, cls="text-xs text-slate-300 font-mono leading-relaxed whitespace-pre-wrap"),
                            cls="p-4 bg-slate-950 border-t border-slate-800"
                        )
                    ),
                    cls="border border-slate-800 rounded-lg overflow-hidden mb-3 bg-slate-900/30 group"
                )
            )

        matches_section = Div(
            H3("Relevant Supreme Court Precedents", cls="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3"),
            Div(*match_components, cls="space-y-1")
        )

        return Div(
            H3("Analysis Complete", cls="text-sm font-bold text-slate-100 mb-4 border-b border-slate-800 pb-2"),
            verdict_card,
            citations_section,
            matches_section
        )

    except Exception as exc:
        logger.error("Analysis proxy failed: %s", exc)
        return Div("Local FastHTML Server proxy timeout or connection error.", cls="text-xs text-red-400 p-4 border border-red-950/20 bg-red-950/15 rounded")

@rt("/explorer-search")
async def post(query: str):
    """Proxies semantic explore searches to FastAPI and renders results."""
    if not query.strip():
        return Div("Please input keywords to explore vector rankings.", cls="text-xs text-slate-400 p-4 border border-slate-800 rounded bg-slate-900/20")
    
    try:
        async with httpx.AsyncClient() as client:
            # We fetch 6 items for the Explorer view to give a wider display
            r = await client.post("http://127.0.0.1:8000/api/v1/analyze", json={"text": query}, timeout=15.0)
        
        if r.status_code != 200:
            return Div("Vector lookup failed.", cls="text-xs text-red-400 p-4 bg-red-950/20 border border-red-900/30 rounded-lg")
        
        res = r.json()
        matches = res.get("semantic_matches", [])

        if not matches:
            return Div("No matches found.", cls="text-xs text-slate-500 italic p-4")

        cards = []
        for match in matches:
            score = match.get("cosine_similarity", 0.0)
            court = match.get("court", "Supreme Court of India")
            pdf = match.get("source_pdf", "Unknown")
            citations_list = ", ".join(match.get("extracted_citations", [])) or "None"
            chunk_id = match.get("chunk_id", "N/A")
            text_snippet = match.get("text_chunk", "").strip()

            cards.append(
                Div(
                    Div(
                        Div(
                            Span(court, cls="text-xs font-semibold text-slate-200"),
                            Span(f"Relevance: {score*100:.1f}%", cls="text-xs font-bold text-amber-400 font-mono"),
                            cls="flex items-center justify-between mb-2"
                        ),
                        Div(
                            Span("Chunk ID: ", cls="text-[10px] text-slate-500"),
                            Span(chunk_id, cls="text-[10px] text-slate-450 font-mono"),
                            cls="mb-3"
                        ),
                        P(f"{text_snippet[:350]}...", cls="text-xs text-slate-400 font-mono leading-relaxed mb-4"),
                        Div(
                            Span("Citations: ", cls="text-[10px] text-slate-500"),
                            Span(citations_list, cls="text-[10px] text-amber-500 font-mono"),
                            cls="border-t border-slate-800 pt-2 flex items-center justify-between"
                        ),
                        cls="p-4"
                    ),
                    cls="bg-slate-900/40 border border-slate-800 rounded-lg hover:border-slate-700 transition-colors"
                )
            )

        return Div(
            Div(
                *cards,
                cls="grid grid-cols-1 md:grid-cols-2 gap-4"
            )
        )

    except Exception as exc:
        logger.error("Explorer search failed: %s", exc)
        return Div("Search request failed.", cls="text-xs text-red-400 p-4 border border-red-950/20 bg-red-950/15 rounded")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local FastHTML Dashboard server on port 8001...")
    uvicorn.run("ui_fasthtml:app", host="127.0.0.1", port=8001, reload=True)
