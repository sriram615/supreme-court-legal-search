(() => {
  "use strict";

  const API_BASE = window.location.origin.includes("null") ? "http://127.0.0.1:8000" : "";
  const ANALYZE_URL = API_BASE + "/api/v1/analyze";
  const HEALTH_URL = API_BASE + "/api/v1/health";
  const HISTORY_KEY = "case-search-console:history";
  const THEME_KEY = "case-search-console:theme";

  const $ = (id) => document.getElementById(id);

  const els = {
    sidebar: $("sidebar"),
    sidebarToggle: $("sidebarToggle"),
    sidebarOpenBtn: $("sidebarOpenBtn"),
    newSearchBtn: $("newSearchBtn"),
    historyList: $("historyList"),
    historyEmpty: $("historyEmpty"),
    clearHistoryBtn: $("clearHistoryBtn"),
    statusDot: $("statusDot"),
    statusLabel: $("statusLabel"),
    themeToggle: $("themeToggle"),
    themeIconSun: $("themeIconSun"),
    themeIconMoon: $("themeIconMoon"),
    transcript: $("transcript"),
    emptyState: $("emptyState"),
    composerForm: $("composerForm"),
    queryInput: $("queryInput"),
    sendBtn: $("sendBtn"),
  };

  /* ───────────────── Theme ───────────────── */
  function applyTheme(mode) {
    if (mode === "light" || mode === "dark") {
      document.documentElement.setAttribute("data-theme", mode);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    const isDark =
      mode === "dark" ||
      (mode !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    els.themeIconSun.style.display = isDark ? "none" : "block";
    els.themeIconMoon.style.display = isDark ? "block" : "none";
  }
  function toggleTheme() {
    const current = localStorage.getItem(THEME_KEY) || "system";
    const isDarkNow =
      current === "dark" ||
      (current === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDarkNow ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
    applyTheme(next);
  }
  try { applyTheme(localStorage.getItem(THEME_KEY) || "system"); } catch (e) { applyTheme("system"); }
  els.themeToggle.addEventListener("click", toggleTheme);

  /* ───────────────── Sidebar ───────────────── */
  function setSidebarCollapsed(collapsed) {
    els.sidebar.classList.toggle("collapsed", collapsed);
  }
  els.sidebarToggle.addEventListener("click", () => setSidebarCollapsed(true));
  els.sidebarOpenBtn.addEventListener("click", () => setSidebarCollapsed(false));

  /* ───────────────── History (localStorage, per-browser) ───────────────── */
  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) { return []; }
  }
  function saveHistory(items) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 50))); } catch (e) {}
  }
  let history = loadHistory();

  function renderHistory(activeId) {
    els.historyList.querySelectorAll(".history-item").forEach((n) => n.remove());
    els.historyEmpty.style.display = history.length ? "none" : "block";
    history.forEach((item) => {
      const btn = document.createElement("button");
      btn.className = "history-item" + (item.id === activeId ? " active" : "");
      btn.textContent = item.query;
      btn.title = item.query;
      btn.addEventListener("click", () => {
        els.queryInput.value = item.query;
        autoGrow();
        els.queryInput.focus();
      });
      els.historyList.appendChild(btn);
    });
  }
  function pushHistory(query) {
    const id = "h_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
    history.unshift({ id, query, ts: Date.now() });
    saveHistory(history);
    renderHistory(id);
    return id;
  }
  els.clearHistoryBtn.addEventListener("click", () => {
    history = [];
    saveHistory(history);
    renderHistory(null);
  });
  renderHistory(null);

  /* ───────────────── New search ───────────────── */
  els.newSearchBtn.addEventListener("click", () => {
    els.transcript.innerHTML = "";
    els.transcript.appendChild(els.emptyState);
    els.emptyState.style.display = "block";
    els.queryInput.value = "";
    autoGrow();
    els.queryInput.focus();
    renderHistory(null);
  });

  /* ───────────────── Example chips ───────────────── */
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.queryInput.value = chip.dataset.example;
      autoGrow();
      submitQuery();
    });
  });

  /* ───────────────── Composer ───────────────── */
  function autoGrow() {
    els.queryInput.style.height = "auto";
    els.queryInput.style.height = Math.min(els.queryInput.scrollHeight, 160) + "px";
  }
  els.queryInput.addEventListener("input", autoGrow);
  els.queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });
  els.composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitQuery();
  });

  /* ───────────────── Health check ───────────────── */
  async function checkHealth() {
    try {
      const res = await fetch(HEALTH_URL);
      const data = await res.json();
      if (res.ok && data.engine_ready) {
        els.statusDot.className = "status-dot ok";
        els.statusLabel.textContent = `Engine ready · ${data.index_vectors_count ?? "?"} vectors`;
      } else {
        els.statusDot.className = "status-dot warn";
        els.statusLabel.textContent = "Engine degraded";
      }
    } catch (e) {
      els.statusDot.className = "status-dot err";
      els.statusLabel.textContent = "Backend unreachable";
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);

  /* ───────────────── Rendering helpers ───────────────── */
  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
  function fmtScore(n) {
    if (typeof n !== "number") return null;
    return n.toFixed(4);
  }

  function ensureTranscriptInner() {
    let inner = els.transcript.querySelector(".transcript-inner");
    if (!inner) {
      els.emptyState.style.display = "none";
      inner = document.createElement("div");
      inner.className = "transcript-inner";
      els.transcript.innerHTML = "";
      els.transcript.appendChild(inner);
    }
    return inner;
  }

  function addUserTurn(query) {
    const inner = ensureTranscriptInner();
    const turn = document.createElement("div");
    turn.className = "turn user-turn";
    turn.innerHTML = `<div class="user-bubble">${esc(query)}</div>`;
    inner.appendChild(turn);
    els.transcript.scrollTop = els.transcript.scrollHeight;
    return turn;
  }

  function addLoadingTurn() {
    const inner = ensureTranscriptInner();
    const turn = document.createElement("div");
    turn.className = "turn assistant-turn";
    turn.innerHTML = `<div class="loading-row">Searching precedent
      <span class="typing-dots"><span></span><span></span><span></span></span></div>`;
    inner.appendChild(turn);
    els.transcript.scrollTop = els.transcript.scrollHeight;
    return turn;
  }

  function renderCitations(resolvedCitations) {
    const entries = Object.entries(resolvedCitations || {});
    if (!entries.length) return "";
    const rows = entries.map(([raw, resolved]) =>
      `<div class="citation-row"><b>${esc(raw)}</b> — ${esc(resolved)}</div>`
    ).join("");
    return `<div class="section-label">Citations found</div><div class="citations-list">${rows}</div>`;
  }

  function renderMatches(matches) {
    if (!matches || !matches.length) {
      return `<div class="section-label">Matching precedent</div>
        <p style="color:var(--text-faint); font-size:13.5px;">No close precedent found for this query.</p>`;
    }
    const cards = matches.map((m, i) => {
      const score = fmtScore(m.rrf_score) ?? fmtScore(m.bm25_score) ?? fmtScore(m.cosine_similarity);
      const scoreLabel = m.rrf_score != null ? "RRF" : m.bm25_score != null ? "BM25" : "cosine";
      const snippet = (m.text_chunk || "").slice(0, 420);
      const cites = (m.extracted_citations || []).slice(0, 6)
        .map((c) => `<span class="cite-tag">${esc(c)}</span>`).join("");
      return `
        <div class="match-card">
          <div class="match-card-head">
            <span class="match-card-title">#${i + 1} · ${esc(m.court || "Unknown court")}</span>
            ${score != null ? `<span class="match-card-score">${scoreLabel} ${score}</span>` : ""}
          </div>
          <div class="match-card-meta">${esc(m.parent_judgment_id || "")} · chunk ${esc(m.chunk_id || "")} · ${esc(m.tokens_count ?? "?")} tokens</div>
          <div class="match-card-snippet">${esc(snippet)}${(m.text_chunk || "").length > 420 ? "…" : ""}</div>
          ${cites ? `<div class="match-card-cites">${cites}</div>` : ""}
        </div>`;
    }).join("");
    return `<div class="section-label">Matching precedent</div>${cards}`;
  }

  function renderVerdict(verdict) {
    if (!verdict) return "";
    // Shape comes from validator.VerificationVerdict:
    // { verdict_agreement: bool (true = contradiction found), legal_rationale: str, confidence_rating: 0..1 }
    if (verdict.available === false) {
      return `
        <div class="section-label">Referee verification</div>
        <div class="verdict-card">
          <span class="verdict-icon" style="color:var(--text-faint)"><svg width="15" height="15" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.4"/><path d="M8 7.5v4M8 5v.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></span>
          <div>
            <div style="font-weight:600; margin-bottom:2px;">Verification unavailable</div>
            <div style="color:var(--text-muted); font-size:13px;">No referee model could be reached, so this result was not checked for contradictions.</div>
          </div>
        </div>`;
    }
    const contradicts = verdict.verdict_agreement === true;
    const baseLabel = contradicts ? "Possible contradiction" : "No contradiction found";
    const conf = typeof verdict.confidence_rating === "number"
      ? ` (${Math.round(verdict.confidence_rating * 100)}% confidence)` : "";
    const label = baseLabel + conf;
    const reasoning = verdict.legal_rationale || "";
    const icon = contradicts
      ? `<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.4"/><path d="M8 4.5v4M8 11v.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`
      : `<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M3 8.5l3 3 7-7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    return `
      <div class="section-label">Referee verification</div>
      <div class="verdict-card">
        <span class="verdict-icon" style="color:${contradicts ? "var(--err)" : "var(--ok)"}">${icon}</span>
        <div>
          <div style="font-weight:600; margin-bottom:2px;">${esc(label)}</div>
          ${reasoning ? `<div style="color:var(--text-muted); font-size:13px;">${esc(reasoning)}</div>` : ""}
        </div>
      </div>`;
  }

  function renderBadges(data) {
    const badges = [];
    // query_type / retrieval_mode are optional fields — only present if the
    // backend has been extended to expose them (see INSTRUCTIONS.md).
    if (data.query_type) {
      badges.push(`<span class="badge intent">${esc(data.query_type)}</span>`);
    }
    badges.push(`<span class="badge mode">${esc(data.retrieval_mode || "Hybrid retrieval")}</span>`);
    return `<div class="badges-row">${badges.join("")}</div>`;
  }

  function replaceTurn(turnEl, html) {
    turnEl.innerHTML = html;
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  /* ───────────────── Submit flow ───────────────── */
  let inFlight = false;
  async function submitQuery() {
    const query = els.queryInput.value.trim();
    if (!query || inFlight) return;

    inFlight = true;
    els.sendBtn.disabled = true;
    els.queryInput.value = "";
    autoGrow();

    addUserTurn(query);
    pushHistory(query);
    const loadingTurn = addLoadingTurn();

    try {
      const res = await fetch(ANALYZE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();

      const html =
        renderBadges(data) +
        renderCitations(data.resolved_citations) +
        renderMatches(data.semantic_matches) +
        renderVerdict(data.verification_verdict);

      replaceTurn(loadingTurn, html);
    } catch (err) {
      replaceTurn(loadingTurn, `<div class="error-row">Search failed: ${esc(err.message || String(err))}</div>`);
    } finally {
      inFlight = false;
      els.sendBtn.disabled = false;
      els.queryInput.focus();
    }
  }
})();
