# Evaluation & Architectural Decisions: Held-Out Retrieval Protocol

> **Document Status:** PROPOSED SPECIFICATION (Approved in Principle; Work Block 1 Foundation).  
> **Repository:** `sriram615/supreme-court-legal-search` (LAWdata)  
> **Scope:** Defines the methodological decisions, empirical evidence, discarded alternatives, governance rules, and work block boundaries for a rigorous, held-out retrieval evaluation.

---

## 1. Approved Protocol Decisions & Technical Rationale

### 1.1 Full-Corpus Evaluation on Active 15,847-Chunk Production Index
* **Decision:** Evaluation queries will search the active production index containing all 15,847 chunks across all 782 judgments located at [`./LAWdata_Corpus_2024/legal_precedents.index`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/legal_precedents.index) and [`./LAWdata_Corpus_2024/metadata_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/metadata_map.json).
* **Rationale:**
  * **Operational Fidelity:** In production, a legal search engine searches the complete case law collection. Evaluating against a pruned sub-corpus evaluates an artificial toy setting, not the actual system.
  * **Distractor Density:** The 217 excluded judgments (`0001`–`0144` and the 73 exposed non-development judgments) provide 5,815 natural legal distractors (hard negatives). Removing them artificially deflates task difficulty and inflates Hit@k metrics.
  * **Corpus Statistics Preservation:** Preserves corpus-wide BM25 Inverse Document Frequency ($IDF$) distributions and FAISS HNSW graph structure without requiring costly or distortive re-indexing.
  * **IR Standard Practice:** Aligns with standard Information Retrieval benchmarks (TREC, MS-MARCO, BEIR), where the document collection is unified and fixed, while test queries and relevance labels are strictly held out.

### 1.2 Absolute Exclusion of 217 Judgments from Query/Target Generation
* **Decision:** Query targets must be sampled strictly from the 565 unexposed candidate judgments (`AWS_SC_2024_0145` through `AWS_SC_2024_0782`, excluding 73 exposed IDs). Zero queries or target annotations may be drawn from the 217 excluded judgments:
  * **144 Development Partition Judgments:** `AWS_SC_2024_0001` through `AWS_SC_2024_0144` (120 of which serve as ground truths in [`data/eval_set.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/eval_set.jsonl), with 24 unselected but inside the partition).
  * **73 Confirmed Exposed Non-Development Judgments:** Deduplicated register comprising 17 tuning/regression targets, 14 precedent label pairs, 39 pure benchmark targets, and 3 diagnostic validation chunks.
* **Rationale:** Prevents direct label leakage, optimization bias, and regression circularity.

### 1.3 Dense Embedding Architecture: `sentence-transformers/all-MiniLM-L6-v2`
* **Decision:** Formalize that dense semantic search uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), executed over FAISS HNSW Flat using inner-product similarity (cosine similarity over L2-normalized vectors).
* **Rationale & Correction:** Reconciles an earlier documentation draft error that erroneously referenced "BGE". Source code verification confirms BGE does not exist in the repository; the engine consistently utilizes `all-MiniLM-L6-v2` ([`search_engine.py:L429, L451`](file:///Users/apple/Desktop/AI-ML/LAWdata/search_engine.py#L429)). The encoder weights are frozen and off-the-shelf, with zero parameter fine-tuning on this corpus.

### 1.4 Graded Multi-Chunk Relevance Schema
* **Decision:** Abandon artificial 1-to-1 query-to-chunk matching in favor of graded, multi-chunk relevance:
  * **Grade 2 (Core Holding / Ratio Decidendi):** Decisive paragraphs articulating the binding rule of law or resolution.
  * **Grade 1 (Supporting Context / Obiter Dicta):** Discussion of relevant statutes, precedents, or factual background.
  * **Grade 0 (Irrelevant / Distractor):** Procedural boilerplate or unrelated legal issues.
* **Rationale:** Legal search queries rarely have only one valid answer chunk in a 50-page judgment. Multi-chunk grading allows calculating true nDCG@k alongside binary Hit@k.

### 1.5 Blind Query Formulation & 3-Gram Anti-Lexical Overlap Rule
* **Decision:** Annotators must formulate queries from factual scenarios and legal issues described in official court headnotes/syllabi *without executing search queries or viewing chunk text*. Queries must not share any 3-gram ($n \ge 3$ consecutive words) with ground-truth chunks, excluding statutory titles and standard legal terms of art.
* **Rationale:** In [`data/eval_set.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/eval_set.jsonl), conceptual queries were created by paraphrasing chunk sentences, causing artificial lexical overlap that inflated BM25 retrieval scores.

### 1.6 Cryptographic Freeze & The Holdout Burning Rule
* **Decision:** The evaluation dataset will be frozen with a SHA-256 seal (`eval/heldout_eval_set_v1.jsonl.sha256`) and executed as a strict **single-run audit**.
* **Governance Rule (Holdout Burning):** If developers inspect evaluation failure modes and modify ranking parameters, heuristics, tokenizers, stopword lists, or boost weights in response, **the evaluation set is immediately declared BURNED and transitions permanently into development data**. Subsequent evaluation claims require sampling a fresh holdout set (`v2`).

### 1.7 Stratified Intent-Specific Metric Reporting
* **Decision:** Retrieval quality must be reported across four stratified intent categories:
  * `CITATION_OR_CASE_NO`: Hit@1, MRR@1 (deterministic routing).
  * `CASE_TITLE`: Hit@1, MRR@1 (party name resolution).
  * `STATUTORY_SECTION`: Hit@3, Hit@5, MRR@5 (statutory chunk isolation).
  * `CONCEPTUAL_OR_AMBIGUOUS`: Hit@5, Hit@10, MRR@5, nDCG@5 (unweighted hybrid RRF fusion).
* **Rationale:** Global macro-averages (e.g., existing Hit@5 = 0.7667) conceal poor conceptual retrieval because citation and case-title shortcuts achieve near-perfect retrieval.

### 1.8 Block 2 Candidate Pool Derivation & Stratified Sampling Specification
* **1.8.1 Exact Candidate Set Derivation:**
  * Total Ingested Corpus: $\mathcal{C}_{\text{total}} = 782$ judgments ($15,847$ chunks).
  * Excluded Pool: $\mathcal{P}_{\text{excluded}} = \mathcal{D}_{\text{dev}} \cup \mathcal{E}_{\text{register}} = 144 + 73 = 217$ judgments ($5,815$ chunks).
  * Candidate Pool: $\mathcal{P}_{\text{candidate}} = \mathcal{C}_{\text{total}} \setminus \mathcal{P}_{\text{excluded}} = 565$ judgments ($10,032$ chunks).
* **1.8.2 Validation by `parent_judgment_id`:**
  * Every chunk in [`metadata_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/metadata_map.json) has an explicit `parent_judgment_id` field matching the `chunk_id` prefix. All candidate chunks and target judgments must be validated strictly against `parent_judgment_id \in \mathcal{P}_{\text{candidate}}` and `parent_judgment_id \notin \mathcal{P}_{\text{excluded}}`.
* **1.8.3 Intent Eligibility Pools & Constrained-First Disjoint Allocation:**
  * The four intent eligibility sub-pools naturally overlap:
    * `CITATION_OR_CASE_NO`: 565 eligible (100% of candidate pool; neutral citation, SCR, Appeal No., or SLP in header).
    * `CASE_TITLE`: 164 eligible (29.0%; clean, unambiguous adversarial `"Petitioner v. Respondent"` in chunk 0).
    * `STATUTORY_SECTION`: 523 eligible (92.6%; mentions statutes, filtered for substantive section interpretation in ratio).
    * `CONCEPTUAL_OR_AMBIGUOUS`: 559 eligible (98.9%; substantive holding length $\ge 1000$ tokens resolving merits).
  * **Reproducible Disjoint Sampling Algorithm:**
    To prevent accidental duplicate target judgments across intents, sampling proceeds in order of increasing candidate availability (most constrained first):
    1. *Stage 1 (`CASE_TITLE`):* Draw target judgments from the 164 title-eligible candidates.
    2. *Stage 2 (`STATUTORY_SECTION`):* Draw target judgments from remaining unassigned candidates in the statutory pool (523 candidates minus Stage 1 selections).
    3. *Stage 3 (`CONCEPTUAL_OR_AMBIGUOUS`):* Draw target judgments from remaining unassigned candidates in the conceptual pool (559 candidates minus Stages 1 & 2 selections).
    4. *Stage 4 (`CITATION_OR_CASE_NO`):* Draw target judgments from all remaining unassigned candidates in the candidate pool.
    *Sampling is deterministic across sorted judgment IDs using a fixed pseudo-random seed (`SEED = 20241015`).*
  * **Intentional Multi-Intent Exception Rule:**
    Single-judgment-per-query is the strict default. If dual-intent evaluation of the same judgment is intentionally planned (e.g. testing cross-modal retrieval of a case via both citation and conceptual prompt), it must be explicitly flagged with `multi_intent_paired: true`, cross-linking both `query_id`s in the audit manifest. Accidental or untracked duplicate targets are strictly prohibited.
* **1.8.4 Sampling Quota & Capacity Planning (Flexible Allocations):**
  * Maximum target: **80 queries**; fallback floor: **60 queries**.
  * **Non-Enforcement of Forced Equal Quotas:** The protocol prohibits manufacturing artificial balance if suitable candidates or annotation capacity do not support it. While an even 20/20/20/20 (or 15/15/15/15) distribution is the initial design target, asymmetric distributions (e.g., 25 Citation, 25 Statutory, 15 Case Title, 15 Conceptual = 80; or 20 Citation, 20 Statutory, 10 Case Title, 10 Conceptual = 60) are explicitly authorized if title clarity or conceptual ratio complexity restricts high-quality manual annotation.
* **1.8.5 Candidate Provenance & Audit Logging:**
  * Every sampled judgment will be logged in an audit manifest recording: `judgment_id`, `source_pdf`, `total_chunks`, `intent_assigned`, `eligibility_criteria_met`, `pseudo_random_seed`, and an automated programmatic assertion confirming exclusion set absence.
* **1.8.6 Proposed Annotation Requirements vs. Reviewer Availability:**
  * *Proposed Protocol Requirement:* Dual independent legal reviewers for relevance grading (Grades 2, 1, 0) preserving separate records, a senior reviewer referee for tie-breaking, and inter-annotator agreement threshold of $\kappa_w \ge 0.70$.
  * *Current Operational Status (Block 3A Reviewer Readiness Assessment):* Human reviewer staffing is **explicitly UNCONFIRMED**. The protocol formalizes the rigorous methodological standard, but all execution of annotation and held-out evaluation (query drafting Block 3B, relevance grading Block 3C, dataset packaging Block 4) remains strictly blocked until qualified human reviewers are confirmed and the human operator explicitly authorizes execution.

### 1.9 Block 3 Blind Annotation Protocol Specification
* **1.9.1 Blind Query-Formulation Instructions by Intent:**
  * `CITATION_OR_CASE_NO`:
    * *Source Input:* Official court header metadata (SCR citation, INSC neutral citation, or Civil/Criminal Appeal/SLP number).
    * *Formulation:* Draft natural citation variations (e.g., `2024 INSC 762`, `[2024] 9 S.C.R. 1`, `Civil Appeal No. 12326 of 2024`).
    * *Prohibition:* Must not contain party names, holding fragments, or factual snippets.
  * `CASE_TITLE`:
    * *Source Input:* Chunk 0 adversarial party names.
    * *Formulation:* Standard attorney title queries (e.g., `Petitioner v. Respondent`, `State of X vs. Y`).
    * *Prohibition:* Must not contain statutory sections, legal issues, or outcome terms.
  * `STATUTORY_SECTION`:
    * *Source Input:* Official headnote/syllabus indicating the substantive section interpreted in ratio.
    * *Formulation:* Name the section and Act alongside the specific legal issue (e.g., `Section 138 Negotiable Instruments Act statutory demand notice period`, `Section 34 Arbitration and Conciliation Act patent illegality`).
    * *Prohibition:* Must not copy the statutory text or judgment phrasing verbatim.
  * `CONCEPTUAL_OR_AMBIGUOUS`:
    * *Source Input:* Factual summary and legal questions from official headnote/syllabus *only*.
    * *Formulation:* Frame as an attorney seeking relevant precedent on a legal controversy without knowing the case name or citation (e.g., `standard for quashing FIR where dispute is purely civil and commercial breach`).
    * *Prohibition:* **Strictly blind to chunk splits and body text.** Annotators are prohibited from viewing judgment chunk text or executing live engine queries.
* **1.9.2 Query Record & Graded Relevance Schema:**
  * The frozen evaluation set must serialize each query to JSONL matching this schema, strictly preserving independent Reviewer A and Reviewer B grades and rationales as separate immutable records, while storing the final adjudicated grade separately:
    ```json
    {
      "query_id": "holdout_eval_001",
      "query": "string (formulated query)",
      "intent_type": "CITATION_OR_CASE_NO | CASE_TITLE | STATUTORY_SECTION | CONCEPTUAL_OR_AMBIGUOUS",
      "target_judgment_id": "AWS_SC_2024_XXXX",
      "source_pdf": "string.pdf",
      "sampling_metadata": {
        "pseudo_random_seed": 20241015,
        "selection_order": 1,
        "eligibility_pool": "string",
        "multi_intent_paired": false
      },
      "graded_relevance": [
        {
          "chunk_id": "AWS_SC_2024_XXXX_chunk_YY",
          "parent_judgment_id": "AWS_SC_2024_XXXX",
          "provenance": "blind syllabus review",
          "review_a": {
            "annotator_id": "REV_A",
            "grade": 2,
            "rationale": "Direct ratio decidendi resolving the statutory interpretation."
          },
          "review_b": {
            "annotator_id": "REV_B",
            "grade": 1,
            "rationale": "Relevant statutory context framing the dispute."
          },
          "adjudication": {
            "final_grade": 2,
            "resolution_type": "consensus",
            "provenance_type": "adjudicated_consensus",
            "adjudicator_id": "REV_SENIOR_01",
            "rationale": "Consensus reached during joint calibration: paragraph articulates binding legal rule."
          }
        }
      ]
    }
    ```
  * *Schema Field Rules & Final-Grade Provenance:*
    * **Preservation of Original Reviewer Records:** `review_a` and `review_b` contain independent, immutable evaluation records (`annotator_id`, `grade` $\in \{0, 1, 2\}$, and `rationale`). Once submitted by the respective reviewer, these fields cannot be modified, overwritten, normalized, or deleted, regardless of the resolution path.
    * `adjudication.final_grade`: The official ground-truth relevance grade used for IR benchmark scoring ($0, 1, 2$).
    * `adjudication.resolution_type`: Allowed values: `"unanimous"`, `"consensus"`, `"referee"`, `"protocol_default"`.
    * `adjudication.provenance_type`: Allowed values: `"copied_agreed_grade"`, `"adjudicated_consensus"`, `"adjudicated_referee"`, `"default_unresolved_consensus"`.
    * **Population of `adjudication.final_grade` and Provenance Distinctions:**
      1. *Unanimous Agreement (`resolution_type: "unanimous"`, `provenance_type: "copied_agreed_grade"`):*
         * *Condition:* `review_a.grade == review_b.grade`.
         * *Population:* `final_grade` is mechanically copied from the common agreed grade (`final_grade = review_a.grade = review_b.grade`).
         * *Provenance Distinction:* This is an unadjudicated, **copied agreed grade** arising from natural independent concordance. `adjudicator_id` is set to `null` (or `"system"`), and `rationale` is logged as `"Natural concordance: unanimous independent evaluation"`.
      2. *Consensus Calibration (`resolution_type: "consensus"`, `provenance_type: "adjudicated_consensus"`):*
          * *Condition:* Adjacent discrepancy (`|review_a.grade - review_b.grade| == 1`, i.e., $2$ vs. $1$).
          * *Population:* `final_grade` is determined through dialogue between Reviewer A and Reviewer B during a joint calibration session facilitated by a designated senior reviewer. The agreed compromise grade ($1$ or $2$) is assigned. If consensus dialogue fails to reach substantive agreement, the unit does NOT remain under consensus; it branches strictly to the separate Protocol Default pathway (Pathway 4 below).
          * *Provenance Distinction:* This is a **deliberative consensus grade determined through joint post-review calibration**. `adjudicator_id` records the senior facilitator ID, and `rationale` documents the agreed legal rationale distinguishing core holding from contextual dicta.
      3. *Referee Determination (`resolution_type: "referee"`, `provenance_type: "adjudicated_referee"`):*
          * *Condition:* Disjunctive discrepancy (`|review_a.grade - review_b.grade| >= 2`, i.e., $2$ or $1$ vs. $0$) or unresolved adjacent impasse.
          * *Population:* `final_grade` is independently determined by the designated Third Senior Reviewer (Referee) ($0, 1, \text{ or } 2$), whose substantive judgment is binding.
          * *Provenance Distinction:* This is an **authoritative grade determined through third-party referee adjudication**. `adjudicator_id` records the specific referee ID (e.g., `"REV_SENIOR_01"`), and `rationale` documents the substantive legal doctrine and textual basis justifying the referee's binding ruling.
      4. *Protocol Default (`resolution_type: "protocol_default"`, `provenance_type: "default_unresolved_consensus"`):*
          * *Condition:* Adjacent discrepancy (`|review_a.grade - review_b.grade| == 1`, i.e., $2$ vs. $1$) where dialogue during the joint calibration session fails to produce substantive consensus between Reviewer A and Reviewer B.
          * *Crucial Methodological Clarification (Not Reviewer Agreement):* **A protocol-default Grade 1 is an administrative holding-purity fallback rule, NOT reviewer agreement.** It does not reflect agreement or concordance between the reviewers, nor does it convert a disagreement into agreement. In the inter-reviewer reliability rating population $\mathcal{U}$, this unit remains a disagreement unit ($n_{12}$ or $n_{21}$) and enters $\kappa_w$ and $AC_2$ as an adjacent discordance.
          * *Preservation of Original Ratings:* Both original ratings ($g_A = 2, g_B = 1$ or vice versa) and their respective legal rationales are **permanently preserved unchanged in `review_a` and `review_b`**.
          * *Population & Provenance Distinction:* This is an **explicit administrative resolution pathway triggered by unresolved consensus dialogue**. `final_grade` is set to $1$ (conservative holding purity to prevent unconfirmed core holding inflation). `adjudicator_id` is set to `"protocol_rule_holding_purity"` (explicitly identifying rule-based adjudication rather than a human reviewer), and `rationale` documents: `"Default Grade 1 applied under holding-purity protocol rule after unresolved consensus dialogue between Reviewer A and Reviewer B."`
* **1.9.3 Reliability Gate: Quadratic-Weighted Cohen's Kappa ($\kappa_w$), Degenerate Fallback & Protocol Governance:**
  * *Unit of Analysis & Evaluation Population for Agreement:*
    * **Distinction Between Query-Level Authoring Quality and Query-Chunk Relevance Grading:**
      * *Query-Level Authoring Quality:* Pertains to the drafted query string $q \in \mathcal{Q}$. Evaluated via automated screening checks (Gate 2.1 whitelist-masked 3-gram filter, token floors, intent formulation instructions Gate 2.2, signed blind isolation declaration Gate 2.3). Authoring is executed by a single blind drafter with qualitative sign-off by the Senior Referee. Query-level authoring does NOT generate paired ordinal grades and DOES NOT enter inter-reviewer reliability statistics ($\kappa_w$ or $AC_2$).
      * *Query-Chunk Relevance Grades:* The candidate units evaluated for inter-reviewer reliability are individual candidate query-chunk pairs $u = (q, c) \in \mathcal{U}$, where $q$ is an approved query and $c$ is a candidate chunk drawn from the parent judgment. Each unit $u$ is independently presented to Reviewer A and Reviewer B without access to the other's grades or to engine retrieval rank. Each reviewer independently assigns an ordinal grade $g \in \{0, 1, 2\}$ and a legal rationale.
      * *Units Entering Reliability Statistics:* **These paired ordinal grades on candidate query-chunk units $u \in \mathcal{U}$ are the sole units of analysis that enter quadratic-weighted Cohen's kappa ($\kappa_w$) and quadratic-weighted Gwet's $AC_2$.**
    * **Set Invariance Across Global and Intent-Stratified Analyses:**
      * Let $\mathcal{U}_{\text{total}}$ be the complete set of all $N_{\text{total}} = |\mathcal{U}_{\text{total}}|$ candidate query-chunk units evaluated by both Reviewer A and Reviewer B across the evaluation set.
      * The set $\mathcal{U}_{\text{total}}$ is partitioned into 4 mutually disjoint, intent-specific subsets based on the query's assigned intent:
        $$\mathcal{U}_{\text{total}} = \mathcal{U}_{\text{citation}} \cup \mathcal{U}_{\text{title}} \cup \mathcal{U}_{\text{statute}} \cup \mathcal{U}_{\text{conceptual}}$$
        where $\mathcal{U}_i \cap \mathcal{U}_j = \emptyset$ for $i \neq j$, and $N_{\text{total}} = \sum_{\text{intent}} N_{\text{intent}}$.
      * *Set Invariance:* The **exact same complete set of independent paired ratings** $(g_A(u), g_B(u))$ collected during the blind review phase is evaluated across both global and stratified analyses.
      * *Global Reliability Analysis:* Evaluates all $N_{\text{total}}$ units pooled together into a single global contingency matrix, computing global statistics ($\kappa_w^{\text{global}}$, $P_{\text{obs}}^{\text{global}}$, $AC_2^{\text{global}}$).
      * *Intent-Stratified Reliability Analysis:* Evaluates each intent subset $\mathcal{U}_{\text{intent}}$ independently, computing stratum-specific statistics ($\kappa_w^{\text{intent}}$, $P_{\text{obs}}^{\text{intent}}$, $AC_2^{\text{intent}}$) over the $N_{\text{intent}}$ units belonging to that stratum.
      * No units are added, removed, re-weighted, or selectively imputed between global and stratified calculations.
  * *Mathematical Formulation of Quadratic-Weighted Cohen's Kappa:*
    * Penalty Matrix ($w_{ij}$): For ordinal categories $i, j \in \{0, 1, 2\}$, quadratic weighting penalizes disagreements proportionally to the square of their category distance:
      $$w_{ij} = \frac{(i - j)^2}{(k - 1)^2} = \frac{(i - j)^2}{(3 - 1)^2} = \frac{(i - j)^2}{4}$$
      Giving weights: $w_{00} = w_{11} = w_{22} = 0.0$; $w_{01} = w_{10} = w_{12} = w_{21} = 0.25$; $w_{02} = w_{20} = 1.0$.
    * Agreement Weights: $a_{ij} = 1 - w_{ij}$.
    * Observed Weighted Agreement ($P_o$):
      $$P_o = \sum_{i=0}^2 \sum_{j=0}^2 a_{ij} \cdot p_{ij}$$
      where $p_{ij} = n_{ij} / N$ is the joint proportion of candidate units assigned grade $i$ by Reviewer A and grade $j$ by Reviewer B.
    * Expected Chance Agreement ($P_e$):
      $$P_e = \sum_{i=0}^2 \sum_{j=0}^2 a_{ij} \cdot (p_{i\cdot} \cdot p_{\cdot j})$$
      where $p_{i\cdot} = \sum_{j} p_{ij}$ and $p_{\cdot j} = \sum_{i} p_{ij}$ are the marginal proportions.
    * Quadratic-Weighted Kappa ($\kappa_w$):
      $$\kappa_w = \frac{P_o - P_e}{1 - P_e} = 1 - \frac{\sum_{i,j} w_{ij} p_{ij}}{\sum_{i,j} w_{ij} (p_{i\cdot} p_{\cdot j})}$$
  * *Policy on Missing Ratings:*
    * **Pairwise deletion and missing-value imputation are strictly prohibited.**
    * Every candidate chunk in the rating population $\mathcal{U}$ must have complete, independent records in both `review_a` and `review_b`.
    * If any chunk lacks a grade or rationale from either reviewer, the dataset fails validation, cannot be processed for reliability statistics, and must be returned to the incomplete reviewer before adjudication proceeds.
  * *Distinguishing Undefined vs. Defined-but-Unstable Weighted Kappa:*
    * **Case A: Undefined Weighted Kappa ($\kappa_w = \text{undefined}$):**
      * Mathematical cause: Occurs when $1 - P_e = 0$ ($P_e = 1.0$), which happens when all $N$ candidate units are assigned to the exact same category by both reviewers (e.g. 100% Grade 0). Both $P_o = 1.0$ and $P_e = 1.0$, producing an indeterminate $\frac{0}{0}$ division by zero. There is zero marginal variance across categories, making kappa mathematically undefined.
    * **Case B: Defined but Potentially Unstable Weighted Kappa (The Marginal Prevalence Paradox):**
      * Mathematical cause: Occurs when $1 - P_e > 0$, so $\kappa_w$ is mathematically well-defined, but $1 - P_e$ is very small ($1 - P_e \le 0.05$, or $P_e \ge 0.95$). This arises from extreme category skew in candidate chunks (e.g., $96\%$ Grade 0, $3\%$ Grade 1, $1\%$ Grade 2). Under such extreme prevalence imbalance, expected chance agreement $P_e$ is exceptionally high, causing the denominator $1 - P_e$ to be near zero. Minor stochastic discrepancies drastically depress $\kappa_w$ to near-zero or negative values despite near-perfect raw concordance ($P_{\text{obs}} \ge 0.98$). The statistic is mathematically defined but psychometrically unstable and misleading.
  * *Fallback Trigger Condition:*
    * The fallback reliability path is triggered if and only if:
      $$\text{Trigger Condition: } (1 - P_e \le 0.05) \quad \lor \quad (\kappa_w \text{ is undefined due to zero marginal variance})$$
    * When $1 - P_e > 0.05$, the rating distribution has adequate marginal spread; the primary gate strictly governs, and the fallback path cannot be invoked.
  * *Canonical Mathematical Formulation of Quadratic-Weighted Gwet's $AC_2$:*
    * **Formal Literature Reference:** Gwet, K. L. (2014). *Handbook of Inter-Rater Reliability: The Definitive Guide to Measuring the Extent of Agreement Among Raters* (4th ed.), Advanced Analytics, LLC. Chapter 4: "Agreement Coefficients for Ordinal, Interval and Ratio Data", Section 4.2: "The AC2 Coefficient", Equation 4.2.3, p. 115.
    * **Validated Implementation References:** Verified against canonical reference implementations: R package `irrCAC` (CRAN) functions `gwet.ac1.table` and `gwet.ac1.raw` with `weights = quadratic.weights(1:q)` (Gwet, 2019/2021), and Python package `irrCAC` (`irrCAC.table.CAC.gwet` and `irrCAC.raw.CAC.gwet` with `weights="quadratic"`, Fergadis, 2020).
    * **Coefficient Formula:**
      $$AC_2 = \frac{P_a^w - P_e^w}{1 - P_e^w}$$
    * **Weight Convention (Fleiss-Cohen Quadratic Weights):** For $K = 3$ ordinal categories ($i, j \in \{0, 1, 2\}$):
      $$w_{ij} = 1 - \frac{(i - j)^2}{(K - 1)^2} = 1 - \frac{(i - j)^2}{(3 - 1)^2} = 1 - \frac{(i - j)^2}{4}$$
      giving symmetric weights: $w_{00} = w_{11} = w_{22} = 1.0$; $w_{01} = w_{10} = w_{12} = w_{21} = 0.75$; $w_{02} = w_{20} = 0.0$.
      The sum of all elements of the $K \times K$ weight matrix $W$ is:
      $$T_w = \sum_{i=0}^{K-1} \sum_{j=0}^{K-1} w_{ij} = 1.0 + 0.75 + 0.0 + 0.75 + 1.0 + 0.75 + 0.0 + 0.75 + 1.0 = 6.0$$
    * **Observed Weighted Agreement Term ($P_a^w$ or $P_o^w$):**
      $$P_a^w = \sum_{i=0}^{K-1} \sum_{j=0}^{K-1} w_{ij} \cdot p_{ij}$$
      where $p_{ij} = n_{ij} / N$ is the joint proportion of candidate units assigned grade $i$ by Reviewer A and grade $j$ by Reviewer B ($N = |\mathcal{U}|$).
    * **Marginal Definitions:**
      * Reviewer A marginal proportion for category $i$: $p_{i\cdot} = \sum_{j=0}^{K-1} p_{ij}$
      * Reviewer B marginal proportion for category $j$: $p_{\cdot j} = \sum_{i=0}^{K-1} p_{ij}$
      * Pooled / average category marginal proportion for category $k$:
        $$\bar{\pi}_k = \frac{p_{k\cdot} + p_{\cdot k}}{2} \quad \text{for } k \in \{0, 1, 2\}$$
    * **Chance-Agreement Term ($P_e^w$):** Under Gwet's AC2 formulation, the weighted chance agreement probability is:
      $$P_e^w = \frac{T_w}{K(K - 1)} \sum_{k=0}^{K-1} \bar{\pi}_k (1 - \bar{\pi}_k)$$
      For $K = 3$ categories and quadratic weights ($T_w = 6.0$, $K(K - 1) = 3 \times 2 = 6.0$):
      $$\frac{T_w}{K(K - 1)} = \frac{6.0}{6.0} = 1.0$$
      Hence, the chance-agreement probability simplifies directly to:
      $$P_e^w = \sum_{k=0}^2 \bar{\pi}_k (1 - \bar{\pi}_k) = 1 - \sum_{k=0}^2 \bar{\pi}_k^2$$
      $AC_2$ adjusts chance agreement based on average category marginals rather than cross-product marginals ($p_{i\cdot} p_{\cdot j}$), preventing the artificial depression of reliability under extreme category prevalence imbalance.
    * **Canonical Worked Reference Benchmark (`cont3x3abstractors`):**
      * *Source Dataset:* Canonical contingency table `cont3x3abstractors` (Gwet 2014, Chapter 3/4; `irrCAC` package), cross-classifying $N = 100$ subjects into $K = 3$ categories by two raters:
        $$\text{table} = \begin{pmatrix} 13 & 0 & 0 \\ 0 & 20 & 7 \\ 0 & 4 & 56 \end{pmatrix}$$
      * *Observed Weighted Agreement:* $P_a^w = \sum_{k,l} w_{kl} p_{kl} = (1.00)(0.13 + 0.20 + 0.56) + (0.75)(0.07 + 0.04) = 0.89 + 0.0825 = 0.9725$.
      * *Rater Marginals & Pooled Marginals:*
        * Rater 1 row marginals: $p_{k\cdot} = [0.13, 0.27, 0.60]$
        * Rater 2 column marginals: $p_{\cdot l} = [0.13, 0.24, 0.63]$
        * Pooled average marginals: $\bar{\pi} = \frac{p_{k\cdot} + p_{\cdot l}}{2} = [0.130, 0.255, 0.615]$ (satisfying $\sum_{k=0}^2 \bar{\pi}_k = 1.000$).
      * *Chance Agreement Calculation:*
        $$P_e^w = 1 - \sum_{k=0}^2 \bar{\pi}_k^2 = 1 - (0.130^2 + 0.255^2 + 0.615^2) = 1 - (0.0169 + 0.065025 + 0.378225) = 1 - 0.46015 = 0.53985$$
        (Equivalently: $\sum_{k=0}^2 \bar{\pi}_k (1 - \bar{\pi}_k) = 0.13(0.87) + 0.255(0.745) + 0.615(0.385) = 0.1131 + 0.189975 + 0.236775 = 0.53985$).
      * *Coefficient Value:*
        $$AC_2 = \frac{0.9725 - 0.53985}{1 - 0.53985} = \frac{0.43265}{0.46015} \approx 0.94024$$
        Replicating the official package docstring outputs for `CAC(data, weights="quadratic").gwet()`.
      * *Arithmetic Reconciliation & Non-Claim of Benchmark Validity:* This worked reference benchmark confirms that the documented mathematical formulas and software implementations are arithmetically reconciled and software-concordant. This arithmetic verification is strictly an algebraic and computational sanity check; it does NOT constitute or claim empirical statistical validity for the held-out evaluation benchmark itself.
    * *Reporting Requirements:* When the fallback is invoked, the audit manifest MUST explicitly publish:
      1. Primary $\kappa_w$ value (or `"UNDEFINED (zero variance)"`),
      2. Full marginal proportions ($p_{i\cdot}, p_{\cdot j}$) and expected chance agreement $P_e$,
      3. Observed Percent Agreement $P_{\text{obs}} = \frac{1}{N} \sum_{i=0}^2 n_{ii}$,
      4. Quadratic-weighted Gwet's $AC_2$,
      5. Explicit documentation of the triggering condition.
  * *Dual Conjunctive Threshold Requirement:*
    * **Both fallback thresholds must pass simultaneously:**
      $$\text{Fallback Gate Pass} \iff (P_{\text{obs}} \ge 0.95) \;\land\; (AC_2 \ge 0.75)$$
    * Passing $P_{\text{obs}} \ge 0.95$ alone is insufficient; passing $AC_2 \ge 0.75$ alone is insufficient. If either falls below its threshold, the fallback gate FAILS.
  * *Status of Thresholds as Proposed Protocol Choices & Statistical Validity Disclaimer:*
    * **Proposed Protocol Choices:** The primary threshold ($\kappa_w \ge 0.70$) and fallback thresholds ($P_{\text{obs}} \ge 0.95$, $AC_2 \ge 0.75$) are **proposed protocol engineering choices adopted for this legal search benchmark**, NOT universal mathematical facts or natural laws. They reflect a conservative policy designed to guarantee high ground-truth label fidelity in a specialized legal domain where core holding boundaries are subtle.
    * **Explicit Non-Claim of Empirical Statistical Validity via Documentation:** Pre-flight documentation verification, code audits, and mathematical formula verifications establish **procedural completeness, internal consistency, and protocol specification correctness, NOT empirical statistical validity**. Empirical statistical validity cannot be established through documentation review; it can only be empirically evaluated on actual annotation data once qualified human reviewers are confirmed and the annotation protocol is executed.
  * *Mandatory Action if Neither Primary Nor Fallback Gate Passes:*
    * If a stratum fails the primary gate ($\kappa_w < 0.70$ when $1 - P_e > 0.05$) OR fails either fallback threshold ($P_{\text{obs}} < 0.95$ or $AC_2 < 0.75$ when triggered):
      1. **Immediate Annotation Halt:** All annotation for the failing intent stratum is immediately stopped.
      2. **Stratum Invalidation:** The tentative query and annotation batch for that stratum is declared invalid and cannot be incorporated into the evaluation set.
      3. **Mandatory Rubric Recalibration Session:** Reviewers A, B, and the Senior Referee must convene to inspect the discrepancy confusion matrix, analyze rationales, and produce a formal written amendment to the grading guidelines clarifying contested holding-vs-context distinctions.
      4. **Mandatory Fresh Sampling / Re-Annotation:** A fresh sample of candidate queries and chunks for that stratum must be drawn and independently re-annotated under the recalibrated rubric (or the full stratum re-graded blindly from scratch) and re-tested against the reliability gates.
* **1.9.4 Three-Gram Overlap Check, Whitelist Span-Masking & Boundary Specification:**
  * *Methodological Status & Explicit Disclaimer:*
    * **Automated Screening Check, NOT Proof of Independent Authorship:** The 3-gram overlap filter is strictly an automated **negative screening filter** designed to catch overt verbatim borrowing from candidate chunk text. Passing the 3-gram check **does NOT constitute affirmative proof of independent conceptual authorship or genuine intellectual drafting**. A query that passes this mechanical check may still exhibit subtle semantic imitation. Affirmative independence relies upon the blind authoring protocol, annotator isolation declarations, and Senior Referee qualitative review.
  * *Step 1: Text Normalization:*
    * Lowercasing: Both query string $Q$ and candidate target chunk text $C$ are lowercased: $Q_{\text{norm}} = \text{lower}(Q)$, $C_{\text{norm}} = \text{lower}(C)$.
    * Punctuation & Whitespace Normalization: All ASCII and Unicode punctuation characters (`[, . ; : " ' ? ! ( ) [ ] { } \ / - _ * ~ `]`) are replaced with single ASCII spaces. Consecutive whitespace characters (spaces, tabs, newlines) are collapsed into a single space, and leading/trailing whitespace is stripped.
  * *Step 2: Tokenization:*
    * Whitespace Tokenization: $Q_{\text{norm}}$ is split on whitespace into an ordered token sequence $T_Q = [q_0, q_1, \dots, q_{m-1}]$; $C_{\text{norm}}$ is split into $T_C = [c_0, c_1, \dots, c_{n-1}]$.
  * *Step 3: Whitelist Matching, Span Overlaps & Sentinel Replacement:*
    * *Whitelist Inventory $\mathcal{W}$:* Closed inventory of canonical multi-token statutory names and doctrinal Latin terms of art.
      * Statutory Patterns: `["section", "<num>"]`, `["article", "<num>"]`, `["order", "<num>"]`, `["rule", "<num>"]`, `["negotiable", "instruments", "act"]`, `["code", "of", "criminal", "procedure"]`, `["code", "of", "civil", "procedure"]`, `["indian", "penal", "code"]`, `["arbitration", "and", "conciliation", "act"]`, `["constitution", "of", "india"]`, `["insolvency", "and", "bankruptcy", "code"]`, `["companies", "act"]`, `["income", "tax", "act"]`, `["motor", "vehicles", "act"]`, etc.
      * Doctrinal Latin Terms of Art: `["prima", "facie"]`, `["res", "judicata"]`, `["ratio", "decidendi"]`, `["obiter", "dicta"]`, `["per", "incuriam"]`, `["audi", "alteram", "partem"]`, `["mens", "rea"]`, `["locus", "standi"]`, `["suo", "motu"]`, `["inter", "alia"]`, `["ex", "parte"]`, `["in", "limine"]`, `["lis", "pendens"]`, `["non", "obstante"]`, etc.
    * *Exact Contiguous Subsequence Matching:* Whitelist phrases are matched against $T_Q$ as exact contiguous token subsequences.
    * *Overlapping Whitelist Matches (Greedy Longest-Match-First):* If multiple whitelist phrases overlap at token index $i$ (e.g., `["negotiable", "instruments", "act"]` vs. `["negotiable", "instruments"]`), the algorithm greedily matches the longest span $[i, j)$ and advances token consumption to index $j$. Sub-matches within an already-consumed token interval are suppressed.
    * *Sentinel Span Masking:* Each matched whitelist token span $T_Q[i:j]$ is replaced by a single atomic sentinel token `__EXEMPT_SPAN__`.
  * *Step 4: Boundary Handling & Preventing Silent Exemption of Surrounding Text:*
    * *Generating Sliding 3-Grams:* Form all sliding 3-grams $G_Q = \{(t_k, t_{k+1}, t_{k+2}) \mid 0 \le k \le |T_Q^{\text{masked}}| - 3\}$ from the masked query tokens.
    * *Exempt Boundary 3-Grams:* Any sliding 3-gram in $G_Q$ containing the sentinel token `__EXEMPT_SPAN__` is an exempt boundary 3-gram ($G_{\text{exempt}}$) and is omitted from chunk collision testing.
    * *Safeguard Against Silent Exemption of Surrounding Text:*
      * Omitting boundary 3-grams that contain `__EXEMPT_SPAN__` ensures legitimate statutory phrases do not trigger false-positive collisions.
      * Non-exempt query tokens occurring *before* the sentinel span form pure 3-grams among themselves; non-exempt query tokens occurring *after* the sentinel span form pure 3-grams among themselves. Both sets are fully tested.
      * Tokens on either side of the sentinel (e.g. $t_{i-1}$ and $t_{j}$) are NEVER joined or stitched together across the sentinel into synthetic 3-grams, as they were not contiguous in the query text.
      * All non-exempt query tokens remain subject to strict 3-gram extraction. Inserting an exempt legal phrase into copied chunk text cannot shield the copied words, as any 3 consecutive non-exempt words remain exposed to collision detection.
  * *Step 5: Short Fragments Containing No Complete 3-Gram:*
    * When a query contains short non-exempt text fragments (e.g., segments between sentinels or a total non-exempt length $< 3$ tokens), sliding 3-grams cannot be formed ($|T_{\text{segment}}| < 3$).
    * To prevent annotators from circumventing the 3-gram check using staccato fragments or ultra-short queries:
      1. *Non-Exempt Token Floor:* Every evaluation query must contain at least 4 non-exempt tokens overall ($|T_Q \setminus \mathcal{W}| \ge 4$). Any query with fewer than 4 non-exempt tokens is automatically rejected during pre-flight validation for insufficient descriptive specification.
      2. *Informational 2-Gram Collision Check:* For any isolated non-exempt 2-token fragment (bigram) adjacent to a sentinel, a secondary 2-gram collision check against target chunk tokens is executed. Verbatim bigram matches do not trigger automatic rejection, but are flagged in the pre-flight audit report, requiring Senior Referee sign-off confirming the bigram is ordinary legal syntax (e.g., `"quash FIR"`) rather than a copied phrase fragment.
  * *Step 6: Target Chunk Collision Testing:*
    * Target chunk 3-grams $G_C = \{(c_l, c_{l+1}, c_{l+2}) \mid 0 \le l \le |T_C| - 3\}$ are extracted from normalized chunk tokens.
    * **Collision Rule:** If any remaining non-exempt query 3-gram intersects target chunk 3-grams ($(G_Q \setminus G_{\text{exempt}}) \cap G_C \neq \emptyset$), the query is **REJECTED**. The annotator must reformulate the query in their own original words.
* **1.9.5 Reviewer Roles, Readiness Assessment & Staffing Blocker (Block 3A):**
  * *Reviewer A and Reviewer B (Primary Dual Reviewers):*
    * **Responsibilities:**
      1. *Blind Query Drafting (Block 3B):* Independently author evaluation queries from official Supreme Court case syllabi/headnotes for assigned candidate judgments across all four retrieval intents (`CITATION_OR_CASE_NO`, `CASE_TITLE`, `STATUTORY_SECTION`, `CONCEPTUAL_OR_AMBIGUOUS`) without accessing candidate chunk text or executing engine searches.
      2. *Independent Relevance Annotation (Block 3C):* Independently inspect candidate query-chunk units $u = (q, c) \in \mathcal{U}$ and assign ordinal relevance grades $g \in \{0, 1, 2\}$ ($2 = \text{Core Holding / Ratio Decidendi}$, $1 = \text{Supporting Context / Obiter Dicta}$, $0 = \text{Irrelevant / Procedural Boilerplate}$) alongside explicit legal rationales.
      3. *Calibration Participation:* Engage in structured joint calibration sessions to resolve adjacent discrepancies ($2$ vs. $1$) through consensus dialogue facilitated by the Senior Referee.
      4. *Blind Isolation Compliance:* Execute and sign the Blind Isolation Declaration (Gate 2.3) certifying zero engine searches, zero prior exposure to retrieval rankings, and zero communication regarding candidate judgments prior to calibration.
    * **Minimum Qualifications:**
      1. Professional law degree (LL.B. or equivalent from a recognized bar-qualifying university).
      2. Demonstrated competency in Indian jurisprudence and Supreme Court case law analysis, specifically the doctrinal capability to distinguish binding legal holdings (*ratio decidendi*) from persuasive or incidental judicial statements (*obiter dicta*).
    * **Independence Requirements:**
      1. *Development & Algorithmic Independence:* Zero involvement in the design, implementation, tuning, or indexing of LAWdata retrieval components (BM25 tokenization, FAISS indexing, heuristic regexes, reciprocal rank fusion multipliers, or UI/CLI search interfaces).
      2. *Search & Exposure Independence:* Zero prior access to engine retrieval scores, rank positions, or development query logs.
      3. *Mutual Reviewer Independence:* Strict blind isolation between Reviewer A and Reviewer B during initial query drafting and primary relevance grading; no sharing or discussion of ratings prior to formal calibration sessions.
    * **Confirmation Evidence:**
      1. Verified legal educational qualification (LL.B. degree or active Bar Council enrollment).
      2. Signed Declaration of Independence and Development Isolation affirming zero contribution to LAWdata retrieval algorithms and zero prior exposure to test targets.
      3. Written confirmation of availability for query authoring, dual annotation, and calibration sessions.
  * *Senior Referee (Third Reviewer / Lead Adjudicator):*
    * **Responsibilities:**
      1. *Pre-Flight Query Oversight:* Review pre-flight audit reports for query quality, including reviewing and signing off on informational 2-gram collision flags adjacent to whitelist sentinels to verify natural legal syntax.
      2. *Calibration Facilitation:* Facilitate joint consensus dialogue between Reviewer A and Reviewer B for adjacent discrepancies ($2$ vs. $1$) under Pathway 2 (`resolution_type: "consensus"`, `provenance_type: "adjudicated_consensus"`).
      3. *Binding Disjunctive Adjudication:* Render binding substantive legal determinations for all disjunctive discrepancies ($2$ or $1$ vs. $0$) and unresolved calibration impasses under Pathway 3 (`resolution_type: "referee"`, `provenance_type: "adjudicated_referee"`), recording legal grounds and adjudicator identity.
      4. *Rubric Recalibration Leadership:* Lead mandatory rubric recalibration and issue formal written amendments if primary or fallback reliability gates fail.
    * **Minimum Qualifications:**
      1. Advanced or senior legal credentials: Master of Laws (LL.M.), Senior Advocate designation, former judicial clerkship (Supreme Court or High Court), recognized law faculty appointment in Indian constitutional/procedural law, or practicing legal advocate with $\ge 5$ years specialized experience in appellate litigation and jurisprudential analysis.
      2. Recognized expertise in Indian Supreme Court precedent analysis, statutory interpretation, and doctrine of precedent (*stare decisis*).
    * **Independence Requirements:**
      1. *Development & Algorithmic Independence:* Complete independence from LAWdata software engineering and retrieval system tuning.
      2. *Neutral Adjudicative Stance:* Neutral third-party status with zero stake or incentive regarding benchmark performance metrics (Hit@k, MRR, NDCG).
      3. *Blind Stance until Adjudication:* Blind to candidate chunk rankings and blinded to initial reviewer identities/grades until formally presented with discrepancy files during adjudication.
    * **Confirmation Evidence:**
      1. Verified senior legal credentials (LL.M. degree, Bar Council enrollment showing $\ge 5$ years practice, judicial clerkship certificate, or faculty appointment record).
      2. Signed Declaration of Adjudicative Neutrality and Development Isolation.
      3. Written confirmation of availability to facilitate calibration sessions and adjudicate discrepancies within the annotation timeframe.
  * *Reviewer Staffing Status (Recorded as UNCONFIRMED):*
    * Reviewer staffing is currently **explicitly UNCONFIRMED**.
    * *Developer Role & Project Context:* The project operator's role is AI engineering developer; reviewer time availability and scheduling capacity are currently unknown. LAWdata is an AI engineering portfolio project demonstrating robust retrieval engineering, hybrid indexing, and rigorous evaluation methodology; it is not a commercial or production legal-services platform.
    * *Zero Fabricated Information:* No reviewer identities, fictitious credentials, or unverified availability schedules are recorded or assumed.
  * *Specification Readiness vs. Execution Readiness:*
    * *Specification Readiness:* **COMPLETE & ACCEPTED.** The comprehensive evaluation methodology, 565-judgment candidate pool, reproducible sampling specification, query formulation rules, separate dual-review and adjudication schema, 3-gram span-masking screening check, and dual-gate reliability framework ($\kappa_w \ge 0.70$; fallback $P_{\text{obs}} \ge 0.95 \land AC_2 \ge 0.75$) are fully accepted as a specification. The independent dual-reviewer protocol and reliability gates are preserved unchanged as the **intended gold-standard evaluation**.
    * *Execution Readiness:* **BLOCKED / NOT READY.** State explicitly that annotation (query drafting and relevance grading) and held-out evaluation remain **STRICTLY BLOCKED** until qualified reviewers are confirmed and the human operator explicitly authorizes execution. No synthetic, developer-authored, or unvalidated queries/annotations may be substituted to bypass the reviewer blocker.
  * *Evaluation Quota Deferral:*
    * The evaluation size quota (80-query target with 20 per intent vs. 60-query fallback with 15 per intent) remains explicitly **deferred** until reviewer availability, capacity, and candidate availability are confirmed.
  * *Product Governance Boundary:*
    * The professional-guidance notice is a separate, lightweight product requirement that is explicitly out of scope for Block 3A and is not implemented in this work block.
* **1.9.6 Reproducible Sampling Procedure (Block 2 Specification):**
  * *Generator:* Python `random.Random(SEED)`.
  * *Deterministic Seed:* `SEED = 20241015`.
  * *Sorting:* Candidate pools sorted lexicographically by `judgment_id` before slicing.
  * *Selection Order:* Constrained-first allocation (Title $\to$ Statute $\to$ Conceptual $\to$ Citation).
  * *Duplicate Prevention:* Set subtraction at each stage ensures strictly disjoint target judgments.

---

## 2. Supporting Repository Evidence vs. Unverified Assumptions

| Item | Status | Exact Source Evidence / Justification |
| :--- | :---: | :--- |
| **Corpus & Index Dimensions** | **Verified Fact** | 782 judgments, 15,847 chunks, 15,847 FAISS vectors ($N=15,847$, 26.6 MB). [`./LAWdata_Corpus_2024/legal_precedents.index`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/legal_precedents.index), [`metadata_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/metadata_map.json). |
| **Development Set Boundary** | **Verified Fact** | 144 judgments (`AWS_SC_2024_0001`–`0144`). 120 queries in [`data/eval_set.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/eval_set.jsonl) map 1-to-1 to 120 judgments in this range; 24 judgments unselected but inside partition. |
| **Contamination Register (73 IDs)** | **Verified Fact** | Deduplicated register: 17 tuning/regression IDs in [`BUILD_REPORT.md`](file:///Users/apple/Desktop/AI-ML/LAWdata/BUILD_REPORT.md) & [`test_and_bench_optimized.py:81`](file:///Users/apple/Desktop/AI-ML/LAWdata/test_and_bench_optimized.py#L81); 14 label IDs in [`eval/precedent_contradiction_labels.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/eval/precedent_contradiction_labels.jsonl); 3 diagnostic IDs in [`validator.py:314-315`](file:///Users/apple/Desktop/AI-ML/LAWdata/validator.py#L314-L315); 46 benchmark targets in [`benchmark_results.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/benchmark_results.json) (7 overlapping). |
| **Candidate Pool Size (565 IDs)** | **Verified Fact** | Set subtraction: $782 - (144 + 73) = 565$ unique `parent_judgment_id` values, covering exactly 10,032 chunks in [`metadata_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/metadata_map.json). |
| **Active Dense Model** | **Verified Fact** | `sentence-transformers/all-MiniLM-L6-v2` explicitly defined at [`search_engine.py:L429`](file:///Users/apple/Desktop/AI-ML/LAWdata/search_engine.py#L429) and [`README.md:L52`](file:///Users/apple/Desktop/AI-ML/LAWdata/README.md#L52). Zero mentions of BGE exist in the codebase. |
| **Stale Corpus Slice Warning** | **Verified Fact** | [`data/processed/aws_court_chunks.jsonl`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/processed/aws_court_chunks.jsonl) contains only 85 chunks (294 KB), an early test artifact. The true corpus is in [`./LAWdata_Corpus_2024/`](file:///Users/apple/Desktop/AI-ML/LAWdata/LAWdata_Corpus_2024/). |
| **Cleanliness of 565 Candidates** | **Unverified Assumption** | Certified only as **absent from recorded repository artifacts**. Unlogged interactive REPL/UI sessions during manual development cannot be audited. |
| **Topological Independence** | **Unverified Assumption** | Assumes reciprocal citation edges between candidate chunks and the 144 dev judgments in [`citation_map.json`](file:///Users/apple/Desktop/AI-ML/LAWdata/data/processed/citation_map.json) do not distort rankings sufficiently to compromise holdout validity. |
| **FAISS CPU Determinism** | **Unverified Assumption** | Assumes FAISS HNSW graph search is strictly deterministic across repeated runs on CPU with fixed thread counts. |

---

## 3. Alternatives Considered and Rejected

### 3.1 Constructing a Separately Filtered 565-Judgment Index
* **Rejected Proposal:** Rebuild FAISS and BM25 indices containing only the 10,032 chunks from the 565 candidate pool, removing all 217 excluded judgments.
* **Why Rejected:**
  * Shrinks search space by 27.7%, reducing distractor competition and unrealistically inflating retrieval metrics.
  * Alters BM25 IDF weights across the corpus, evaluating a model that behaves differently from production.
  * Violates standard IR benchmarking methodology (where corpus is fixed and queries are held out).

### 3.2 Using the 85-Chunk File in `data/processed/`
* **Rejected Proposal:** Point evaluation or index loading to `data/processed/aws_court_chunks.jsonl`.
* **Why Rejected:** Source inspection confirmed this file is an obsolete 85-chunk stub (294 KB). Using it would result in a broken evaluation.

### 3.3 Binary Single-Chunk Target Matching
* **Rejected Proposal:** Forcing every evaluation query to map to exactly one positive `chunk_id`.
* **Why Rejected:** Real legal reasoning spans multiple paragraphs across holding and contextual analysis. Single-chunk constraints produce false-negative evaluation errors when the engine retrieves an equally valid adjacent chunk.

### 3.4 Iterative Holdout Tuning
* **Rejected Proposal:** Allowing developers to inspect holdout failure queries, tune section regexes or RRF weights, and re-run evaluation.
* **Why Rejected:** Tuning against evaluation queries converts the holdout set into a training/validation set, re-introducing the exact contamination that invalidated `eval_set.jsonl`.

---

## 4. Known Limitations & Remaining Approvals

### 4.1 Epistemological Limitations
1. **Absence of Proof is Not Proof of Cleanliness:** The 565 candidate judgments are clean *with respect to recorded repository artifacts*. If a developer queried an unrecorded candidate case in the FastHTML UI during exploration, residual human tuning bias may exist.
2. **Citation Graph Leakage:** The engine boosts chunks citing authoritative precedents. If candidate judgments cite development judgments, reciprocal graph edges in `citation_map.json` inject relevance boosts derived from the development corpus.

### 4.2 Status of Decisions & Block 3A Reviewer Readiness Assessment
1. **Candidate Pool Specification (Block 2):** **APPROVED IN PRINCIPLE BY OPERATOR.**
2. **Blind Annotation Specification (Block 3 Methodology & Statistics):** **SPECIFICATION REVIEW COMPLETE (Accepted by Operator).** Methodology and statistical specification accepted as a specification, subject to recorded limitations. Final-grade provenance distinguished (copied agreed grade, consensus, referee adjudication, and explicit protocol-default holding purity); 3-gram overlap refined with greedy longest-match, boundary non-stitching, token floor, and screening disclaimer; reliability gate unit of analysis defined across global and stratified sets; canonical quadratic-weighted Gwet's $AC_2$ formula verified against Gwet (2014) and `irrCAC`; worked numerical reference benchmark arithmetically reconciled without claiming empirical benchmark validity; thresholds preserved as proposed protocol choices.
3. **Reviewer Readiness Assessment (Block 3A):** **DOCUMENTATION COMPLETE.** Responsibilities, minimum qualifications, independence requirements, and confirmation evidence fully specified for Reviewer A, Reviewer B, and Senior Referee. Staffing recorded as UNCONFIRMED (developer role, portfolio project context, reviewer time availability unknown). Specification readiness distinguished from execution readiness: the independent dual-reviewer protocol and reliability gates are preserved unchanged as the intended gold-standard evaluation, while execution remains strictly blocked pending confirmed qualified reviewers and explicit operator authorization. Evaluation quota remains deferred. Professional-guidance notice noted as a separate, lightweight product requirement not implemented in this block.
4. **Preserved Unresolved Decisions (Awaiting Human Decision):**
   * **Evaluation Size Quota:** 80-query target (20 per intent) vs. 60-query fallback (15 per intent), deferred until reviewer availability and candidate availability are confirmed.
   * **Statutory Whitelist Scope:** Final confirmation of whether the multi-token statutory and Latin term-of-art inventory $\mathcal{W}$ remains strictly closed as defined or allows controlled pre-annotation expansion to domain-specific statutes identified during candidate review.
   * **Reviewer Staffing Confirmation:** Confirmation and qualification verification of specific legal reviewers (Reviewer A, Reviewer B, Senior Referee with verified LL.B./senior credentials and Supreme Court jurisprudence competency).
5. **Execution Blocker Before Annotation Execution:**
   * Reviewer staffing is currently **explicitly unconfirmed**. Under protocol governance, query formulation (Block 3B), annotation collection (Block 3C), dataset packaging (Block 4), and benchmark execution (Block 5) are **STRICTLY BLOCKED** pending formal confirmation of qualified reviewers and explicit human authorization to begin execution.
   * Block 3B, Block 3C, and Block 4 are **NOT AUTHORIZED**.

---

## 5. Work Blocks and Authorization Boundaries

| Work Block | Name | Scope & Deliverables | Authorization Status |
| :---: | :--- | :--- | :---: |
| **Block 1** | **Documentation Foundation** | Create `decisions.md` and `flow.md`. Document architecture, protocol, and boundaries. No code or dataset changes. | **COMPLETED (Authorized)** |
| **Block 2** | **Candidate Pool Specification** | Programmatic derivation of 565 candidate judgments ($10,032$ chunks). Intent eligibility distributions, `parent_judgment_id` validation, up-to-80 quota design. | **APPROVED BY OPERATOR** |
| **Block 3A** | **Reviewer Readiness Assessment** | Document responsibilities, qualifications, independence, and confirmation evidence for Reviewers A/B and Referee; record unconfirmed staffing; distinguish specification vs. execution readiness; defer quota; preserve gold standard. | **COMPLETED DOCUMENTATION (Accepted; Execution Blocked: Reviewers Unconfirmed)** |
| **Block 3B** | **Blind Query Formulation** | Author 60–80 evaluation queries across 4 intents strictly blind to chunks; 3-gram negative screening filter; isolation declarations. | **BLOCKED / NOT AUTHORIZED (Requires Confirmed Reviewers & Operator Authorization)** |
| **Block 3C** | **Dual Relevance Annotation** | Independent dual-reviewer chunk grading ($0/1/2$), separate consensus/referee adjudication schema, reliability gates ($\kappa_w \ge 0.70$, $P_{\text{obs}} \ge 0.95 \land AC_2 \ge 0.75$). | **BLOCKED / NOT AUTHORIZED (Requires Confirmed Reviewers & Operator Authorization)** |
| **Block 4** | **Packaging, Gates & Sealing** | Pre-flight 3-gram checks, parent ID exclusion checks, formatting `heldout_eval_set_v1.jsonl`, generating SHA-256 seal. | **BLOCKED (Requires Human Approval; Block 4 Not Authorized)** |
| **Block 5** | **Single-Run Benchmark Execution** | One-shot execution against active index (`./LAWdata_Corpus_2024/`, $N=15,847$). Automated run-validity checks (infra bug vs. miss). | **BLOCKED (Requires Human Approval)** |
| **Block 6** | **Stratified Analysis & Governance** | Generate stratified intent reports, compare with baseline, enforce holdout burning governance if code is modified. | **BLOCKED (Requires Human Approval)** |

---

## 6. Portfolio Evaluation Tracks (Proposed Specification)

> **Document Status:** PROPOSED SPECIFICATION (Separate Portfolio Track).  
> **Relationship to Gold-Standard Protocol:** These portfolio evaluation tracks are exploratory AI engineering portfolio benchmarks, strictly separate from the accepted gold-standard legal-review protocol (Blocks 1–6). The dual-reviewer legal annotation protocol remains preserved as the intended gold standard, currently blocked pending confirmed qualified legal reviewers.

To demonstrate retrieval capabilities within the operational realities of an AI engineering portfolio project (where qualified legal reviewers are unconfirmed and developer time availability is unknown), two separate portfolio evaluation tracks are proposed:

### 6.1 Track A — Known-Answer Retrieval
* **Core Purpose:** Test whether known case identifiers or citations retrieve their expected judgments.
* **Scope:** Focuses on deterministic and semi-structured retrieval pathways (e.g., standard case citations, appeal numbers, party names, and statutory provisions where unambiguous ground-truth target judgments exist).
* **Role in Portfolio:** Validates query intent classification, structured inverted indexing, exact-identifier filtering, and lexical/dense recall against known historical targets without requiring deep jurisprudential ratio annotation.

### 6.2 Track B — Conceptual Retrieval
* **Core Purpose:** Explore semantic retrieval using conceptual queries, with the provenance and limitations of relevance labels explicitly recorded.
* **Scope:** Explores dense semantic and unweighted hybrid RRF retrieval for thematic, doctrinal, or natural-language attorney inquiries.
* **Provenance & Limitations Notice:** Because qualified legal reviewers are unconfirmed, any exploratory relevance labels developed for Track B must explicitly record their provenance (e.g., author, source, derivation methodology) and epistemological limitations. Such labels do NOT claim gold-standard legal validity or replacement of the formal dual-reviewer protocol.

*(Note: Detailed metric definitions, candidate sampling rules, query generation procedures, and execution implementations are deliberately not designed at this stage and remain deferred to subsequent steps.)*

