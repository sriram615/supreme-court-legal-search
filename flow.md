# Retrieval Architecture & Evaluation Workflow: Held-Out Protocol

> **Document Status:** PROPOSED SPECIFICATION (Approved in Principle; Work Block 1 Foundation).  
> **Repository:** `sriram615/supreme-court-legal-search` (LAWdata)  
> **Scope:** Details the runtime search engine architecture, existing evaluation baseline, end-to-end held-out evaluation workflow, validation gates, failure paths, and execution governance.

---

## 1. Current Retrieval Architecture & Baseline

### 1.1 Production Search Pipeline Architecture

```
                               [User Query]
                                     |
                                     v
                       [Query Classification Node]
                     classify_and_extract_query(query)
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    [Dense Semantic Channel]                    [Lexical Channel]
  all-MiniLM-L6-v2 (384-dim)                  BM25Okapi (k1=1.5, b=0.75)
  FAISS HNSW Flat Index                       15,847 Pre-Tokenized Chunks
  Inner-Product Cosine Similarity              Lexical Keyword Scores
  (search_engine.py:884-899)                  (search_engine.py:901-904)
               |                                           |
         FAISS Ranks (r_f)                           BM25 Ranks (r_b)
               |                                           |
               +---------------------+---------------------+
                                     |
                                     v
                 [Structured Inverted Identifier Filter]
                 citation_index, section_index, title_index
                 Exact-Identifier Matches -> 2.5x RRF Multiplier
                 Conceptual / Ambiguous  -> 1.0x (Unweighted)
                                     |
                                     v
                    [Reciprocal Rank Fusion (RRF)]
                     RRF Score = Sum(weight / (60 + rank))
                                     |
                                     v
                      [Ranked Top-k Search Results]
```

#### Detailed Execution Steps ([`search_engine.py:L846-L980`](file:///Users/apple/Desktop/AI-ML/LAWdata/search_engine.py#L846-L980)):
1. **Query Intent Classification:** Analyzes query syntax using regex heuristics to categorize into `CITATION_OR_CASE_NO`, `CASE_TITLE`, `STATUTORY_SECTION`, or `CONCEPTUAL_OR_AMBIGUOUS`.
2. **Dense Vector Search:** Encodes query using `sentence-transformers/all-MiniLM-L6-v2`. Searches the 15,847-vector FAISS HNSW Flat graph (`efSearch=64`), generating candidate vector ranks ($r_{\text{FAISS}}$) and cosine similarities.
3. **Lexical Keyword Search:** Tokenizes query and computes lexical match scores across 15,847 chunks via `BM25Okapi`.
4. **Structured Inverted Identifier Lookup:** Queries precomputed inverted dictionaries (`citation_index`, `section_index`, `title_index`). For structured query types, exact identifier matches receive priority candidate ordering and a $2.5\times$ RRF weight multiplier. For `CONCEPTUAL_OR_AMBIGUOUS` queries, this filter is bypassed and weight remains $1.0\times$ (unweighted).
5. **Reciprocal Rank Fusion (RRF, $k=60$):** Combines dense and lexical ranks:
   $$\text{RRF Score} = \frac{1}{60 + r_{\text{FAISS}}} + \frac{w}{60 + r_{\text{BM25}}}$$
   Results are sorted descending by RRF score to produce the final top-$k$ hits.

### 1.2 Active Production Index vs. Stale Test Artifacts

* **Active Index Path:** [`./LAWdata_Corpus_2024/legal_precedents.index`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/legal_precedents.index) (26.6 MB) and [`./LAWdata_Corpus_2024/metadata_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/metadata_map.json) (51.4 MB). Contains exactly 15,847 chunks across all 782 judgments.
* **Corpus Chunks Source:** [`./LAWdata_Corpus_2024/aws_court_chunks.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/aws_court_chunks.jsonl) (51.2 MB, 15,847 lines).
* **Warning (Stale Artifact):** [`data/processed/aws_court_chunks.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/processed/aws_court_chunks.jsonl) contains only 85 chunks (294 KB) from early pipeline tests and must never be referenced by the evaluation suite.

### 1.3 Existing Evaluation Baseline (Contaminated Development Set)

The existing evaluation suite operates on [`data/eval_set.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/eval_set.jsonl) via [`eval/run_retrieval_eval.py`](file:///Users/apple/Desktop/AI-ML/LAWdata/eval/run_retrieval_eval.py):
* **Dataset Size:** 120 queries (30 per intent category: Citation, Case Title, Statute, Conceptual).
* **Ground Truth Boundary:** 100% of ground-truth targets fall within `AWS_SC_2024_0001` through `AWS_SC_2024_0144` (the initial development partition).
* **Current Reported Baseline:**
  * **Global Hit@5:** $0.7667$ ($92 / 120$ queries successful).
  * **Global MRR@5:** $0.6492$.
* **Known Deficiencies of Baseline:**
  1. *Development Contamination:* Queries and targets were used to tune ranking heuristics in [`BUILD_REPORT.md`](file:///Users/apple/Desktop/AI-ML/LAWdata/BUILD_REPORT.md).
  2. *Synthetic Overlap:* Conceptual queries directly copy phrases from target chunk text.
  3. *Artificial Metric Masking:* 100% Hit@1 on citation queries masks degraded conceptual retrieval performance.

---

## 2. Proposed Held-Out Evaluation Workflow

```
[Candidate Extraction]
  - Filter metadata_map.json by parent_judgment_id NOT IN Excluded_217
  - Yields 565 Candidate Judgments (10,032 chunks)
        |
        v
[Domain Stratification & Sampling]
  - Stratify into Constitutional, Criminal, Commercial, Civil
  - Sample 60-80 Judgments for Target Set
        |
        v
[Blind Query Annotation]
  - Review official syllabus/headnotes only
  - Zero search execution / No chunk text viewing
  - Enforce 3-gram anti-lexical overlap filter
        |
        v
[Dual Graded Relevance Grading]
  - Review candidate chunks for sampled judgments
  - Assign Grade 2 (Holding), Grade 1 (Context), Grade 0 (Irrelevant)
  - Dual review records & separate adjudication; measure quadratic-weighted Cohen's Kappa (kappa_w >= 0.70)
        |
        v
[Packaging & Validation Gates]
  - Verify parent_judgment_id NOT IN Excluded_217
  - Verify query parsing and anti-overlap rules
  - Generate eval/heldout_eval_set_v1.jsonl & SHA-256 seal
        |
        v
[One-Shot Benchmark Execution]
  - Run once against active production index (15,847 chunks)
  - Verify execution completeness (infra bug vs. retrieval miss)
        |
        v
[Stratified Reporting & Holdout Governance]
  - Report metrics separated by intent
  - If code is modified in response to defects -> RETIRE/BURN HOLDOUT
```

---

## 3. Workflow Steps: Inputs, Outputs, and Validation Gates

### Step 1: Candidate Pool Derivation & Stratified Sampling Specification
* **Inputs:**
  * Active metadata store: [`./LAWdata_Corpus_2024/metadata_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/metadata_map.json) ($15,847$ chunks, 782 parent judgments).
  * Excluded Pool Register ($217$ unique IDs): $144$ development partition judgments (`AWS_SC_2024_0001`–`0144`) + $73$ confirmed exposure IDs from deduplicated register.
* **Outputs:**
  * Candidate Pool: Exactly **565 unique candidate judgments** comprising **10,032 chunks**.
  * Sampling Manifest: Specifications for up to 80 target judgments (aiming for 20 per intent where natural candidate pools support it, with fallback floor of 60).
* **Validation Gate 1.1 (Set Invariance):**
  * Assert $|\mathcal{C}_{\text{total}}| = 782$.
  * Assert $|\mathcal{P}_{\text{excluded}}| = 217$.
  * Assert $|\mathcal{P}_{\text{candidate}}| = 782 - 217 = 565$.
  * Assert $\mathcal{P}_{\text{candidate}} \cap \mathcal{P}_{\text{excluded}} = \emptyset$.
* **Validation Gate 1.2 (`parent_judgment_id` Integrity):**
  * Assert every chunk in `metadata_map.json` has `parent_judgment_id == chunk_id.rsplit('_chunk_', 1)[0]`.
  * Assert exactly $10,032$ chunks have `parent_judgment_id \in \mathcal{P}_{\text{candidate}}`.
* **Validation Gate 1.3 (Constrained-First Disjoint Sampling & Natural Eligibility):**
  * To prevent duplicate targets across overlapping pools, sampling executes in order of increasing candidate availability using pseudo-random seed `SEED = 20241015`:
    1. `CASE_TITLE`: Allocated first from 164 judgments (29.0%) possessing clean adversarial party headers (`"v."` / `"versus"`) in chunk 0.
    2. `STATUTORY_SECTION`: Allocated second from unassigned judgments in the statutory pool (523 candidates with statutory sections, filtered for ratio-level interpretation).
    3. `CONCEPTUAL_OR_AMBIGUOUS`: Allocated third from unassigned judgments in the conceptual pool (559 candidates with substantive holding length $\ge 1000$ tokens).
    4. `CITATION_OR_CASE_NO`: Allocated fourth from all remaining unassigned candidates (all 565 candidates qualify with neutral INSC, SCR, Appeal No., or SLP headers).
  * **Intentional Multi-Intent Pairing:** Single judgment per query is the strict default. If dual-intent evaluation of the same judgment is intentionally planned, it must be explicitly flagged with `multi_intent_paired: true`, linking both `query_id`s in the audit manifest. Accidental duplication is prohibited.
* **Validation Gate 1.4 (Quota Flexibility):**
  * Upper bound: **80 queries**; fallback floor: **60 queries**.
  * Forced equal quotas (20/20/20/20 or 15/15/15/15) are strictly prohibited if suitable candidates or annotation capacity are constrained; asymmetric distributions are explicitly authorized.
* **Status:** **COMPLETED SPECIFICATION (Pending Human Approval)**.

### Step 2: Blind Query Formulation
* **Inputs:** Official case headnotes/syllabi for sampled candidate judgments.
* **Outputs:** 60–80 evaluation queries across four intents (`CITATION_OR_CASE_NO`, `CASE_TITLE`, `STATUTORY_SECTION`, `CONCEPTUAL_OR_AMBIGUOUS`).
* **Validation Gate 2.1 (Anti-Lexical Overlap & Whitelist Span-Masking Gate):**
  * Automated pre-flight check rejecting any query sharing $n \ge 3$ consecutive words with target chunk text.
  * *Methodological Status Disclaimer:* This is strictly an automated **negative screening filter** to catch overt verbatim copying, **NOT proof of independent authorship or genuine intellectual drafting**. Affirmative independence relies on blind authoring declarations and referee qualitative oversight.
  * *Text Normalization:* Lowercase string, convert all punctuation characters to single spaces, collapse whitespace, and trim.
  * *Tokenization:* Split on whitespace into ordered token stream.
  * *Legal-Term Whitelist:* Closed dictionary of canonical multi-token statutory names (e.g., `["section", "<num>"]`, `["negotiable", "instruments", "act"]`, `["code", "of", "criminal", "procedure"]`) and Latin terms of art (`["prima", "facie"]`, `["res", "judicata"]`, `["ratio", "decidendi"]`, `["per", "incuriam"]`).
  * *Span Matching & Overlap Resolution:* Contiguous token subsequence matching; overlapping whitelist matches are resolved via greedy longest-match-first. Matched spans are replaced by atomic sentinel `__EXEMPT_SPAN__`.
  * *Boundary Handling & Non-Exempt Filtering:* Sliding 3-grams are generated. Boundary 3-grams containing `__EXEMPT_SPAN__` are excluded from testing. Non-exempt tokens on either side form 3-grams within their own segments and are strictly tested against normalized chunk 3-grams; tokens are NEVER joined across sentinels into synthetic 3-grams. If $(G_Q \setminus G_{\text{exempt}}) \cap G_C \neq \emptyset$, the query is rejected.
  * *Short Fragments (< 3 Tokens) & Non-Exempt Token Floor:* Every query must satisfy a non-exempt token floor of at least 4 tokens overall ($|T_Q \setminus \mathcal{W}| \ge 4$); queries with $< 4$ non-exempt tokens fail pre-flight validation. Any isolated 2-token non-exempt fragment adjacent to a sentinel undergoes an informational 2-gram check, requiring Senior Referee sign-off confirming it is natural syntax rather than a copied phrase fragment.
* **Validation Gate 2.2 (Intent Formulation Instructions):**
  * `CITATION_OR_CASE_NO`: Natural variants from official metadata without factual fragments.
  * `CASE_TITLE`: Standard adversarial party queries without statutory or holding tokens.
  * `STATUTORY_SECTION`: Section + Act alongside substantive legal issue, avoiding verbatim statutory reproduction.
  * `CONCEPTUAL_OR_AMBIGUOUS`: Drafted as attorney inquiries solely from headnote legal controversies; strictly blind to chunk text and engine output.
* **Validation Gate 2.3 (Blind Verification):** Annotators sign an isolation declaration affirming zero engine searches were executed during drafting.
* **Preserved Unresolved Decisions:**
  * Evaluation size quota: 80-query target (20 per intent) vs. 60-query fallback (15 per intent), deferred until reviewer availability and candidate availability are confirmed.
  * Statutory whitelist scope: whether the multi-token statutory and Latin term inventory $\mathcal{W}$ remains strictly closed as defined or allows controlled pre-annotation expansion to domain-specific statutes.
  * Reviewer staffing: qualified reviewers unconfirmed.
* **Dependency & Staffing Blocker:** Reviewer staffing is **explicitly unconfirmed** (Block 3A Reviewer Readiness Assessment completed; Block 3B query formulation blocked). Under protocol governance, Block 3B query formulation, Block 3C annotation collection, dataset creation, and evaluation execution are **STRICTLY BLOCKED** pending formal confirmation of qualified reviewers and explicit human authorization to begin execution.
* **Status:** **SPECIFICATION COMPLETE (Execution Blocked: Reviewers Unconfirmed & Human Authorization Required; Block 3B Not Authorized)**.

### Step 3: Dual Graded Relevance Annotation
* **Inputs:** Candidate chunks from the sampled judgments.
* **Outputs:** Graded relevance mapping and query record JSONL schema preserving independent Reviewer A and Reviewer B records, with separate adjudication:
  * Schema structure per query record:
    * `query_id`, `query`, `intent_type`, `target_judgment_id`, `source_pdf`, `sampling_metadata`.
    * `graded_relevance` array of chunk records containing:
      * `chunk_id`, `parent_judgment_id`, `provenance`.
      * `review_a`: `{"annotator_id": string, "grade": 0|1|2, "rationale": string}`.
      * `review_b`: `{"annotator_id": string, "grade": 0|1|2, "rationale": string}`.
      * `adjudication`: `{"final_grade": 0|1|2, "resolution_type": "unanimous"|"consensus"|"referee"|"protocol_default", "provenance_type": "copied_agreed_grade"|"adjudicated_consensus"|"adjudicated_referee"|"default_unresolved_consensus", "adjudicator_id": string|null, "rationale": string}`.
  * Grade Definitions:
    * Grade 2 = Core Holding / Ratio Decidendi.
    * Grade 1 = Supporting Context / Obiter Dicta.
    * Grade 0 = Irrelevant / Procedural Boilerplate.
  * Final-Grade Population & Provenance Distinctions:
    * Original `review_a` and `review_b` records are immutable and permanently preserved unchanged.
    * *Unanimous Agreement (`resolution_type: "unanimous"`, `provenance_type: "copied_agreed_grade"`):* `review_a.grade == review_b.grade`; mechanically copied from common agreed grade (`final_grade = review_a.grade`), with `adjudicator_id: null`.
    * *Consensus Calibration (`resolution_type: "consensus"`, `provenance_type: "adjudicated_consensus"`):* Adjacent discrepancy ($2$ vs. $1$) resolved through dialogue during calibration facilitated by a senior reviewer; `final_grade` determined through consensus compromise ($1$ or $2$), with `adjudicator_id` recording the senior facilitator ID. If consensus dialogue fails to reach substantive agreement, the unit branches strictly to the separate Protocol Default pathway.
    * *Referee Determination (`resolution_type: "referee"`, `provenance_type: "adjudicated_referee"`):* Disjunctive discrepancy ($2$ or $1$ vs. $0$) or unresolvable impasse; `final_grade` independently determined by Third Senior Reviewer (Referee), with `adjudicator_id` recording referee ID and rationale recording legal grounds.
    * *Protocol Default (`resolution_type: "protocol_default"`, `provenance_type: "default_unresolved_consensus"`):* Adjacent discrepancy ($2$ vs. $1$) where dialogue fails to reach consensus. **A protocol-default Grade 1 is an administrative holding-purity fallback rule, NOT reviewer agreement.** Both original ratings in `review_a` and `review_b` are permanently preserved unchanged (retaining discordance in the reliability rating population $\mathcal{U}$). `final_grade = 1`, `adjudicator_id = "protocol_rule_holding_purity"`, and `rationale` documents the holding-purity fallback rule.
* **Validation Gate 3.1 (Reliability Gate: Quadratic-Weighted Cohen's Kappa, Fallback & Governance):**
  * *Unit of Analysis & Population Definition:*
    * *Distinction:* Query-level authoring quality is screened via automated filters (Gate 2.1 3-gram filter, token floors, intent formulation instructions Gate 2.2, signed blind declaration Gate 2.3) and does NOT enter inter-reviewer reliability statistics. The units of analysis entering $\kappa_w$ and $AC_2$ are strictly the candidate query-chunk pairs $u = (q, c) \in \mathcal{U}$ independently evaluated by Reviewer A and Reviewer B.
    * *Set Invariance:* The complete paired ratings set $\mathcal{U}_{\text{total}}$ ($N_{\text{total}}$ units) is partitioned into 4 mutually disjoint intent subsets ($\mathcal{U}_{\text{citation}}$, $\mathcal{U}_{\text{title}}$, $\mathcal{U}_{\text{statute}}$, $\mathcal{U}_{\text{conceptual}}$). The exact same complete set of independent paired ratings is evaluated pooled in the global analysis and independently in the intent-stratified analyses, without adding, dropping, or imputing units.
  * *Mathematical Formulation of Quadratic-Weighted Cohen's Kappa:* Quadratic penalty weights $w_{ij} = (i - j)^2 / 4$ for ordinal categories $i, j \in \{0, 1, 2\}$, agreement weights $a_{ij} = 1 - w_{ij}$, observed weighted agreement $P_o = \sum_{i,j} a_{ij} p_{ij}$, chance agreement $P_e = \sum_{i,j} a_{ij} (p_{i\cdot} p_{\cdot j})$, yielding $\kappa_w = (P_o - P_e) / (1 - P_e)$.
  * *Missing Ratings Policy:* Pairwise deletion and missing-value imputation are strictly prohibited. Every candidate chunk must have complete, independent `review_a` and `review_b` records before agreement calculation.
  * *Undefined vs. Defined-but-Unstable Kappa:*
    * *Undefined $\kappa_w$:* Occurs when $1 - P_e = 0$ ($P_e = 1.0$), caused by zero marginal variance (100% of units assigned to a single category by both reviewers), yielding indeterminate $0/0$.
    * *Defined but Unstable $\kappa_w$ (Prevalence Paradox):* Occurs when $1 - P_e > 0$ but $1 - P_e \le 0.05$ ($P_e \ge 0.95$), caused by extreme category skew. Minor discrepancies depress $\kappa_w$ toward zero or negative values despite high raw concordance ($P_{\text{obs}} \ge 0.98$).
  * *Fallback Trigger Condition:* Triggered if and only if $1 - P_e \le 0.05$ OR $\kappa_w$ is undefined due to zero marginal variance. When $1 - P_e > 0.05$, the primary gate ($\kappa_w \ge 0.70$) strictly governs.
  * *Canonical Mathematical Formulation of Quadratic-Weighted Gwet's $AC_2$:*
    * *Literature Reference:* Gwet, K. L. (2014), *Handbook of Inter-Rater Reliability: The Definitive Guide to Measuring the Extent of Agreement Among Raters* (4th ed.), Advanced Analytics, LLC, Chapter 4, Section 4.2, Eq. 4.2.3. Validated implementation: R package `irrCAC` (`gwet.ac1.table` / `gwet.ac1.raw` with `weights = quadratic.weights(1:q)`) and Python package `irrCAC` (`CAC.gwet` with `weights="quadratic"`).
    * *Formula:* $AC_2 = (P_a^w - P_e^w) / (1 - P_e^w)$.
    * *Observed Weighted Agreement:* $P_a^w = \sum_{i,j} w_{ij} p_{ij}$, using Fleiss-Cohen quadratic weights $w_{ij} = 1 - (i - j)^2 / 4$ ($w_{00}=w_{11}=w_{22}=1.0$, $w_{01}=w_{10}=w_{12}=w_{21}=0.75$, $w_{02}=w_{20}=0.0$, total weight sum $T_w = 6.0$).
    * *Marginals:* Reviewer marginals $p_{i\cdot} = \sum_j p_{ij}$, $p_{\cdot j} = \sum_i p_{ij}$; pooled category marginals $\bar{\pi}_k = (p_{k\cdot} + p_{\cdot k}) / 2$.
    * *Chance Agreement:* $P_e^w = \frac{T_w}{K(K - 1)} \sum_{k=0}^{K-1} \bar{\pi}_k (1 - \bar{\pi}_k)$. For $K = 3$ and quadratic weights ($T_w = 6.0$, $K(K-1) = 6$), $\frac{T_w}{K(K-1)} = 1.0$, so $P_e^w = \sum_{k=0}^2 \bar{\pi}_k (1 - \bar{\pi}_k) = 1 - \sum_{k=0}^2 \bar{\pi}_k^2$.
    * *Canonical Worked Benchmark:* Reconciled against canonical dataset `cont3x3abstractors` ($N=100$, row marginals $p_{k\cdot} = [0.13, 0.27, 0.60]$, col marginals $p_{\cdot l} = [0.13, 0.24, 0.63]$, pooled marginals $\bar{\pi} = [0.130, 0.255, 0.615]$, $P_a^w = 0.9725$, $P_e^w = 0.53985$, $AC_2 = 0.94024$). Arithmetically reconciled against reference implementations without claiming empirical benchmark validity.
    * *Audit Reporting:* Publish primary $\kappa_w$, marginals, $P_e$, $P_{\text{obs}}$, quadratic-weighted $AC_2$, and triggering condition in the audit manifest.
  * *Dual Conjunctive Threshold Requirement:* **Both fallback thresholds must pass simultaneously:** $P_{\text{obs}} \ge 0.95$ AND quadratic-weighted $AC_2 \ge 0.75$. If either fails, the fallback gate fails.
  * *Protocol Choice Status & Statistical Validity Disclaimer:*
    * Thresholds ($\kappa_w \ge 0.70$, $P_{\text{obs}} \ge 0.95$, $AC_2 \ge 0.75$) are **proposed protocol engineering choices adopted for this benchmark**, not universal mathematical facts.
    * **Explicit Non-Claim Disclaimer:** Documentation checks and mathematical verifications establish **procedural completeness and protocol specification consistency, NOT empirical statistical validity**. Empirical statistical validity can only be evaluated on actual empirical annotation data once qualified reviewers are confirmed and annotations are collected.
  * *Mandatory Failure Action:* If a stratum fails the primary gate (when $1 - P_e > 0.05$) OR fails either fallback threshold (when triggered), annotation for that stratum is immediately halted, tentative annotations are invalidated, reviewers and referee must execute a mandatory rubric recalibration session, and a fresh candidate sample must be independently annotated from scratch.
* **Validation Gate 3.2 (Disagreement Resolution & Referee Adjudication):**
  * Adjacent ($2$ vs. $1$): Joint calibration session to reach consensus; if unresolved, default conservatively to Grade $1$ under the separate Protocol Default pathway (holding purity).
  * Disjunctive ($2$ or $1$ vs. $0$): Adjudicated by a designated **Third Senior Reviewer (Referee)** whose independent decision is final and logged under `adjudication.final_grade`, `adjudication.adjudicator_id`, and `adjudication.rationale`.
* **Dependency & Reviewer Readiness Assessment (Block 3A Completed; Reviewer Staffing Unconfirmed):**
  * *Reviewer A and Reviewer B (Primary Dual Reviewers):*
    * Responsibilities: Blind query authoring from official headnotes across all four intents (Block 3B); independent grading of candidate units $u = (q, c) \in \mathcal{U}$ on $0/1/2$ ordinal scale with written rationales (Block 3C); participation in joint consensus calibration for adjacent discrepancies ($2$ vs. $1$); signing blind isolation declarations (Gate 2.3).
    * Minimum Qualifications: Professional law degree (LL.B. or equivalent from a recognized bar-qualifying university); demonstrated competency in Indian Supreme Court jurisprudence, specifically distinguishing *ratio decidendi* (binding rule) from *obiter dicta* (contextual dicta).
    * Independence Requirements: Zero involvement in LAWdata retrieval engine development, tuning, or indexing; zero prior access to retrieval scores, rank positions, or query logs; strict mutual blind isolation between Reviewers A and B during initial drafting and grading.
    * Confirmation Evidence: Verified law degree or active Bar Council enrollment; signed Declaration of Independence & Development Isolation; written confirmation of availability.
  * *Senior Referee (Third Reviewer / Lead Adjudicator):*
    * Responsibilities: Pre-flight query review and sign-off on informational 2-gram collision flags adjacent to whitelist sentinels; neutral procedural facilitation of consensus calibration sessions; binding substantive legal determination for disjunctive discrepancies ($2$ or $1$ vs. $0$) and unresolved calibration impasses under Pathway 3 (`resolution_type: "referee"`); leading rubric recalibration sessions upon reliability gate failure.
    * Minimum Qualifications: Senior legal credentials (LL.M., Senior Advocate designation, former judicial clerkship, recognized law faculty appointment, or practicing advocate with $\ge 5$ years specialized appellate litigation experience in Indian jurisprudence).
    * Independence Requirements: Full development/algorithmic independence; neutral third-party status with no incentive regarding benchmark recall metrics; adjudicative neutrality blind to initial reviews until referee adjudication.
    * Confirmation Evidence: Verified senior credentials (LL.M. degree, Bar Council enrollment showing $\ge 5$ years practice, judicial clerkship certificate, or faculty appointment record); signed Declaration of Adjudicative Neutrality; written confirmation of availability.
  * *Staffing Status (Recorded as UNCONFIRMED):*
    * Reviewer staffing is currently **explicitly UNCONFIRMED**.
    * Developer role & portfolio project context: The project operator's role is AI engineering developer; reviewer time availability and scheduling capacity are currently unknown. LAWdata is an AI engineering portfolio project demonstrating robust retrieval engineering and evaluation methodology, not a production legal-services platform.
    * Zero fabricated information: No reviewer identities, credentials, or schedules are invented.
  * *Specification Readiness vs. Execution Readiness:*
    * Specification Readiness: **COMPLETE & ACCEPTED.** The independent dual-reviewer protocol, separate adjudication schema, and reliability gates ($\kappa_w \ge 0.70$, fallback $P_{\text{obs}} \ge 0.95 \land AC_2 \ge 0.75$) are preserved unchanged as the **intended gold-standard evaluation**.
    * Execution Readiness: **BLOCKED / NOT READY.** State explicitly that annotation (query drafting and relevance grading) and held-out evaluation remain **STRICTLY BLOCKED** until qualified reviewers are confirmed and the human operator explicitly authorizes execution.
  * *Evaluation Quota Deferral:* Evaluation quota (80-query target vs. 60-query fallback) remains deferred until reviewer availability is known.
  * *Product Governance Boundary:* The professional-guidance notice is a separate, lightweight product requirement that is explicitly out of scope for Block 3A and is not implemented in this work block.
* **Status:** **BLOCK 3A COMPLETED (Specification Accepted; Execution Blocked: Reviewers Unconfirmed & Human Authorization Required; Block 3C Not Authorized)**.

### Step 4: Packaging, Pre-Flight Gates & Cryptographic Sealing
* **Inputs:** Verified queries, graded relevance labels, active metadata map.
* **Outputs:** `eval/heldout_eval_set_v1.jsonl` and `eval/heldout_eval_set_v1.sha256`.
* **Validation Gate 4.1 (Zero Target Contamination):** Automated pre-commit script verifies that every ground-truth `chunk_id` belongs to a `parent_judgment_id` in the 565 candidate pool and $\notin$ the 217 excluded judgments.
* **Validation Gate 4.2 (Query Syntax):** Verify every query successfully parses through `classify_and_extract_query()` without unhandled exceptions.
* **Status:** **PROPOSED (Blocked pending human approval)**.

### Step 5: One-Shot Benchmark Execution & Run-Validity Filtering
* **Inputs:** Frozen dataset, active production index ([`./LAWdata_Corpus_2024/legal_precedents.index`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/legal_precedents.index), $N=15,847$).
* **Outputs:** Raw evaluation logs and metric summaries.
* **Validation Gate 5.1 (Run Validity vs. Infrastructure Failure):**
  * *Infrastructure Failure (Run Discarded):* Any unhandled Python exception, index corruption, memory abort, or returned result list with length $< k$. The run is voided; holdout remains unburned.
  * *Legitimate Retrieval Miss (Valid Evaluation):* Engine returns top-$k$ chunks, but annotated target is absent. Scored as $\text{Hit@}k = 0$, $\text{MRR@}k = 0$. Factored into aggregate metrics.
* **Status:** **PROPOSED (Blocked pending human approval)**.

### Step 6: Stratified Analysis, Reporting & Holdout Governance
* **Inputs:** Evaluation results, baseline metrics from `data/eval_set.jsonl`.
* **Outputs:** Published evaluation report displaying stratified intent metrics alongside macro-averages.
* **Validation Gate 6.1 (Holdout Retirement Rule):**
  * If developers inspect failure queries and modify ranking heuristics, regexes, stopword lists, boost multipliers, or RRF parameters in response:
  * **The evaluation dataset is declared BURNED and immediately transitions to development data.**
  * Subsequent holdout claims require annotating a fresh holdout set (`v2`).
* **Status:** **PROPOSED (Blocked pending human approval)**.

---

## 4. Current Work Block Status Summary

| Work Block | Description | Execution Status | Authorization |
| :---: | :--- | :---: | :---: |
| **Block 1** | **Documentation Foundation** (`decisions.md`, `flow.md`) | **COMPLETED** | **Authorized** |
| **Block 2** | **Candidate Pool & Sampling Specification** | **COMPLETED SPECIFICATION** | **Approved by Operator** |
| **Block 3A** | **Reviewer Readiness Assessment** (Documentation) | **COMPLETED DOCUMENTATION** | **Accepted as Specification (Execution Blocked: Reviewers Unconfirmed)** |
| **Block 3B** | **Blind Query Formulation** | **BLOCKED** | **Requires Confirmed Reviewers & Operator Authorization (Not Authorized)** |
| **Block 3C** | **Dual Graded Relevance Annotation** | **BLOCKED** | **Requires Confirmed Reviewers & Operator Authorization (Not Authorized)** |
| **Block 4** | **Packaging, Gates & Cryptographic Sealing** | **PROPOSED** | **Awaiting Human Approval (Not Authorized)** |
| **Block 5** | **One-Shot Benchmark Execution** | **PROPOSED** | **Awaiting Human Approval** |
| **Block 6** | **Stratified Reporting & Holdout Governance** | **PROPOSED** | **Awaiting Human Approval** |

---

## 5. Portfolio Evaluation Tracks (Proposed Specification)

> **Document Status:** PROPOSED SPECIFICATION (Separate Portfolio Track).  
> **Relationship to Gold-Standard Protocol:** These portfolio evaluation tracks are exploratory AI engineering portfolio benchmarks, strictly separate from the accepted gold-standard legal-review protocol (Steps 1–6). The dual-reviewer legal annotation protocol remains preserved as the intended gold standard, currently blocked pending confirmed qualified legal reviewers.

To demonstrate retrieval capabilities within the operational realities of an AI engineering portfolio project (where qualified legal reviewers are unconfirmed and developer time availability is unknown), two separate portfolio evaluation tracks are proposed:

### 5.1 Track A — Known-Answer Retrieval
* **Core Purpose:** Test whether known case identifiers or citations retrieve their expected judgments.
* **Scope:** Focuses on deterministic and semi-structured retrieval pathways (e.g., standard case citations, appeal numbers, party names, and statutory provisions where unambiguous ground-truth target judgments exist).
* **Role in Portfolio:** Validates query intent classification, structured inverted indexing, exact-identifier filtering, and lexical/dense recall against known historical targets without requiring deep jurisprudential ratio annotation.

### 5.2 Track B — Conceptual Retrieval
* **Core Purpose:** Explore semantic retrieval using conceptual queries, with the provenance and limitations of relevance labels explicitly recorded.
* **Scope:** Explores dense semantic and unweighted hybrid RRF retrieval for thematic, doctrinal, or natural-language attorney inquiries.
* **Provenance & Limitations Notice:** Because qualified legal reviewers are unconfirmed, any exploratory relevance labels developed for Track B must explicitly record their provenance (e.g., author, source, derivation methodology) and epistemological limitations. Such labels do NOT claim gold-standard legal validity or replacement of the formal dual-reviewer protocol.

*(Note: Detailed metric definitions, candidate sampling rules, query generation procedures, and execution implementations are deliberately not designed at this stage and remain deferred to subsequent steps.)*

