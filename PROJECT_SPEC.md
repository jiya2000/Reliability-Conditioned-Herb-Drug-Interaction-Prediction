# PROJECT_SPEC.md: Reliability-Conditioned Herb–Drug Interaction (HDI) Prediction

**Project Redesign & Specification Document**  
*Date:* October 28, 2026  
*Status:* Active Blueprint  

---

## 1. What the Project Becomes

### The Shift in Thesis
* **Old Claim:** A deep multimodal fusion model (Heterogeneous GAT + BiomedBERT) predicts herb–drug interactions with superior accuracy compared to prior models.
* **New Claim:** Literature-derived HDI benchmarks are heavily dominated by superficial shortcuts (outcome words in abstracts yield AUC 0.97; entity identity memorization yields AUC 0.80). Rather than training deep models on leaky text, we explicitly decouple scientific evidence from ground truth. We learn which published studies to trust using an agreement-supervised reliability model, aggregate multi-study findings with citation echo-discounting, collapse naturally to a signed mechanistic PK prior when literature is thin, and selectively abstain when evidence is conflicting or uncertain.

This redesign transitions the codebase to a smaller, interpretable model with a clean scientific argument where every subcomponent is rigorously falsifiable and auditable.

---

## 2. Two Distinct Prediction Tasks

To eliminate cross-task contamination and evaluation ambiguity, the benchmark formally distinguishes between two clinical prediction settings:

| Dimension | Task A: Cold-Start Interaction Prediction | Task B: Evidence Aggregation & Verdict Verification |
| :--- | :--- | :--- |
| **Clinical Question** | *Is there likely a pharmacokinetic interaction for a botanical-drug pair with zero published clinical literature?* | *Given a body of heterogeneous, potentially conflicting studies on this pair, what is the true clinical verdict?* |
| **Model Inputs** | Signed mechanistic & pharmacokinetic features only (enzyme/transporter modulation, $f_m$, clearance fraction, dose, reachability $\rho$). | Published study findings (positive/negative/inconclusive) + study design/quality descriptors + mechanistic prior from Task A. |
| **Target Label** | Curated Consensus Label (from external clinical reference compendia). | Curated Consensus Label (external ground truth independent of study texts). |
| **Evaluation Splits** | Herb-disjoint, drug-disjoint, and pair-disjoint grouped cross-validation. | Pair-disjoint grouped by citation cluster. |

---

## 3. Data Architecture (Three Auditable Tables)

### Table 1: Studies Table (Existing Evidence Records)
One row per individual publication/trial report (`data/benchmark/verified_botanical_hdi.csv`):
* **Identifiers:** `source_id` (PMID / DOI), `study_id`, `herb_id`, `drug_id`, `cited_study_ids`.
* **Parsed Study Design Fields:** `species` (human, in vitro, rat), `assay_type`, `route` (oral, IV), `preparation` (whole crude, ethanolic extract, isolated constituent), `dose`, sample size $n$, `standardized_extract` (yes/no, marker content).
* **Finding (Input, NOT Label):** The primary study's finding (`interaction_present` / `no_interaction_observed` / `inconclusive`). 
* **Key Constraint:** Individual study findings are now **model features**, never target labels.

### Table 2: Consensus Labels Table (`data/benchmark/consensus_labels.csv`)
One row per unique candidate pair `(herb_id, drug_id)`:
* **Target Classes:** `interaction` / `no_interaction` / `disputed`.
* **Clinical Metadata:** Severity class (major, moderate, minor) and effect direction (increased exposure, decreased exposure) where established.
* **Curated External Sources:** NaPDI (National PDI Center), Natural Medicines Comprehensive Database, Stockley's Drug Interactions, DrugBank Clinical Interactions, and FDA CYP/Transporter Clinical Tables.
* **Ground Truth Rules:**
  1. The label must originate strictly from external curated compendia, never from the abstract text of papers inside the Studies table.
  2. Annotation must be performed blind to model predictions.
  3. Double-annotate at least a 20% validation subset to report inter-annotator agreement (Cohen's $\kappa$).
  4. **Rule for `disputed`:** Assigned when authoritative clinical sources disagree, or when evidence is explicitly noted as conflicting (e.g., Ginseng + Digoxin, Ginkgo + Warfarin).

### Table 3: Mechanistic Knowledge Table (Rebuilt from Per-Value Sources)
Replaces hardcoded heuristic dictionaries with a citation-grounded database:
* **Per Herb:** Specific constituent modulation values for CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2, P-gp (ABCB1), and OATP1B1 (SLCO1B1). Each entry specifies modulation direction (induce, inhibit, null), potency class ($K_i$ / $\text{IC}_{50}$ or qualitative), and a verifiable citation (DOI/PMID).
* **Per Drug:** Fractional metabolism contribution ($f_m$) for each CYP pathway, P-gp substrate status, and Narrow Therapeutic Index (NTI) clinical classification, with corresponding clinical pharmacology citations.

### Scale Targets
* **Existing Benchmark:** 141 verified publication records covering **120 unique (herb, drug) pairs** across 18 botanicals and 40 drugs.
* **Target Scale:** Expand consensus annotations to **250–400 pairs** (utilizing candidate pairings from `consensus_labels_random_unstudied.csv` and curated external compendia).

---

## 4. Features

### 1. Mechanistic Pair Features (Task A & Task B Prior)
Sign-aware biological interaction features (enzyme induction decreases drug exposure; enzyme inhibition increases exposure):
$$\text{signed\_effect}_e = \text{herb\_effect}_e \times \text{drug\_}f_{m,e}$$
* Computed for each CYP enzyme ($e \in \{\text{3A4, 2C9, 2C19, 2D6, 1A2}\}$) and P-glycoprotein.
* Summary aggregations: $\max_e |\text{signed\_effect}_e|$, $\sum_e |\text{signed\_effect}_e|$, and NTI flag ($1.0$ if narrow therapeutic index).
* Tissue exposure compartmentalization: Hepatic vs. intestinal reachability flags.
* **Strict Privacy Rule:** Herb and drug entity identities (one-hot IDs, string names) are **never** used as features.

### 2. Study Design & Quality Features (For Reliability Model $R$)
Descriptors capturing study rigor and clinical translatability:
* Human clinical in vivo ($1.0$), human ex vivo ($0.5$), animal in vivo ($0.25$), in vitro cell assay ($0.05$).
* Route of administration (oral match $= 1.0$).
* Preparation purity (standardized extract $= 1.0$, crude powdered leaf $= 0.3$).
* Reported dosage and sample size $n$.
* **Pharmacokinetic Reachability Parameter $\rho$:**
  * Hepatic: $\rho_{\text{hepatic}} = \min(1.0, [I]_{u,\max} / \text{IC}_{50})$
  * Intestinal (CYP3A4 / P-gp): $I_{\text{gut}} = \text{Dose} / 250\text{ mL}$

### 3. Citation Lineage Features
* **Echo Status:** Primary trial report ($e_i = 1.0$) vs. secondary citation echo ($e_i = \alpha$, where $\alpha < 1.0$).
* **Study Genre:** Double-blind randomized controlled trial (RCT), prospective cohort, case report, narrative review.
* **Citation Cluster Size:** Number of derivative publications repeating the same primary data.

### 4. Text Representation Policy
* **Never Raw Text:** Raw text abstracts are completely excluded from valid models to prevent outcome-word leakage.
* Text is used solely for automated extraction of design metadata (via audited regex/parsers) or in outcome- and entity-masked benchmarking diagnostic baselines.

---

## 5. Model Architecture

```
                    [ Mechanistic Knowledge Table ]
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Task A: Mechanistic Prior │
                    │   (Logistic Regression)   │
                    └─────────────┬─────────────┘
                                  │ logit_prior
                                  ▼
 [ Study Design Fields ]    ┌───────────┐
           │                │ R Model   │ ──> w_i = R_i · e_i
 [ Citation Lineage DAG ] ──>│ (Agreem.) │
                            └───────────┘
                                  │
                                  ▼
               [ Evidence Aggregator: S_pos, S_neg ]
                                  │
                                  ▼ Net Evidence E
             ┌─────────────────────────────────────────┐
             │ logit(p) = logit_prior + λ · E          │
             │ C = 2·min(S_pos, S_neg) / (S_pos+S_neg) │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
                 [ Conformal Abstention Layer ]
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
     [ INTERACTION ]       [ NO INTERACTION ]      [ ABSTAIN ]
                                               (Pharmacist Review)
```

### 1. Task A: Mechanistic Prior
Logistic regression fit on signed mechanistic features:
$$\text{logit}_{\text{prior}} = \beta_0 + \sum_e \beta_e \cdot \text{signed\_effect}_e + \beta_{\text{NTI}} \cdot \text{NTI}$$
A gradient boosted tree (HistGradientBoosting) serves as a non-linear comparison.

### 2. Supervised Reliability Model $R_i$
Trained on studies with clear findings evaluated against authoritative consensus labels:
* Agreement indicator: $a_i = 1$ if $\text{finding}_i == \text{consensus\_label}$, else $0$.
* Supervised reliability:
$$R_i = \sigma(g(\text{design}_i, \text{lineage}_i))$$
* Trained with Binary Cross-Entropy (BCE) under grouped CV.
* Evaluated by ROC-AUC and Expected Calibration Error (ECE) for predicting study truthfulness.

### 3. Residual Evidence Aggregation (Task B)
For a pair with $N$ published studies:
1. **Echo-Discounted Weight:** $w_i = R_i \cdot e_i$, where $e_i = 1.0$ for primary studies and $\alpha$ for derivative echoes ($\alpha \in [0, 1]$ learned/tuned in CV).
2. **Polarity Mass:**
   $$S_{\text{pos}} = \sum_{i: \text{finding}=\text{pos}} w_i, \quad S_{\text{neg}} = \sum_{i: \text{finding}=\text{neg}} w_i$$
3. **Net Evidence Signal:**
   $$E = \frac{S_{\text{pos}} - S_{\text{neg}}}{S_{\text{pos}} + S_{\text{neg}} + \kappa}$$
   where $\kappa > 0$ is a Bayesian evidence shrinkage prior.
4. **Residual Formulation:**
   $$\text{logit}(p) = \text{logit}_{\text{prior}} + \lambda \cdot E, \quad (\lambda \ge 0)$$
   * **Collapse-to-Prior Guarantee:** If no literature exists ($N=0$), $S_{\text{pos}} = S_{\text{neg}} = 0 \implies E = 0$, causing $\text{logit}(p) \equiv \text{logit}_{\text{prior}}$ by mathematical construction.
5. **Contradiction Index $C$:**
   $$C = \frac{2 \cdot \min(S_{\text{pos}}, S_{\text{neg}})}{S_{\text{pos}} + S_{\text{neg}} + \epsilon}$$

---

## 6. Calibration, Conformal Inference & Abstention

1. **Pooled Out-of-Fold Calibration:** Temperature scaling parameters are fit on pooled out-of-fold validation predictions across CV folds, never on small arbitrary sub-splits.
2. **Tri-State Clinical Decision Support:**
   $$\hat{Y} \in \{\text{INTERACTION},\, \text{NO INTERACTION},\, \text{ABSTAIN}\}$$
   Abstentions route cases directly to clinical pharmacist review (advisory clinical support).
3. **Split-Conformal Prediction Sets:** Conformal calibration over out-of-fold non-conformity scores. Abstain if:
   * The conformal prediction set contains both classes $\{0, 1\}$, or
   * Contradiction index $C > \tau_c$, or
   * Epistemic confidence falls below threshold.
4. **Falsification Test on Disputed Pairs:** The abstention framework is validated against the subset of pairs clinically cataloged as `disputed`. A valid reliability and contradiction mechanism must abstain at a significantly higher rate on disputed pairs than naive confidence heuristics.

---

## 7. Evaluation Protocol & Rigor

### Primary Splits
5×5 Repeated Grouped Cross-Validation:
* **Herb-Disjoint Split:** All candidate pairs involving test herbs are withheld from training.
* **Drug-Disjoint Split:** All candidate pairs involving test drugs are withheld from training.
* **Pair-Disjoint (Cluster-Grouped) Split:** Candidate pairs and their citation clusters are strictly segregated.

### Required Comparison Baselines
1. **Shortcut Diagnostics (Marked `LEAKY REFERENCE (not a valid model)`):**
   * Herb-identity memorization (one-hot).
   * Drug-identity memorization (one-hot).
   * Raw abstract TF-IDF / unmasked text models.
2. **Simple Baselines:**
   * Unsupervised mechanistic rule baseline (shared pathway heuristic).
   * Tabular Logistic Regression & HistGradientBoosting.
3. **Aggregation Baselines:**
   * Unweighted majority vote.
   * Quality-weighted vote ($R_i = \text{mean\_source\_quality}$, $\alpha = 1$).
   * Echo-discounted vote (fixed heuristic $R_i$, $\alpha = 0.15$).

### Ablation Matrix
All ablation variants must receive identical clean quality-only inputs:

| Ablation Variant | Hypothesis / Architectural Test |
| :--- | :--- |
| **Prior Only** | Quantifies the standalone predictive power of pharmacokinetic mechanisms without literature. |
| **Evidence Only ($\lambda \to \infty$)** | Quantifies performance when ignoring biological mechanism and relying solely on reported literature. |
| **$R = 1$ (No Reliability)** | Tests whether supervised study reliability weighting improves upon simple evidence counting. |
| **$\alpha = 1$ (No Echo Discounting)** | Tests whether discounting redundant derivative citations prevents echo-chamber false consensus. |
| **Random $R$** | Permutation control verifying that learned weights reflect genuine study rigor rather than variance. |
| **No Design Features in $R$** | Feature ablation isolating the exact contribution of species, route, extract type, and dose. |

### Evaluation Slices & Statistics
* **Performance Slices:**
  1. *Hard Negatives:* Strong theoretical mechanistic overlap but verified negative clinical trials (e.g., in vitro CYP3A4 inhibition failing in vivo reachability).
  2. *Sparse vs. Rich Evidence:* Pairs with 1 study vs. pairs with $\ge 4$ independent studies.
* **Statistical Metrics:** ROC-AUC, PR-AUC, F1, Equal-Mass ECE, Brier Score, Continuous AURC, and 95% bootstrap confidence intervals. Paired differences reported with Holm-Bonferroni correction.
* **Permanent Test Invariants:**
  * Rule R1: Zero outcome language or polarity features reach any model in the main table.
  * Rule R2: Disjoint splitting enforces zero entity/cluster overlap between training and test sets.
  * Rule R3: Any non-reference model exceeding ROC-AUC 0.98 raises an automated alert/failure.
  * Rule R4: Every numerical decimal in reports must be deterministically grounded in `results/*.json`.

---

## 8. Repository Structure & Refactoring Plan

| Component | Status | Action / Architectural Change |
| :--- | :--- | :--- |
| **Source Registry & Lineage** | **Keep** | Retain strict registry verification (`require_registered=True`) and citation DAG clustering. |
| **PK Reachability Modules** | **Keep** | Retain gut and hepatic reachability calculations ($\rho_{\text{hepatic}}, \rho_{\text{gut}}$). |
| **Conformal & Calibration** | **Keep** | Retain temperature scaling and split-conformal inference engine. |
| **Mechanistic Profiles** | **Change** | Rebuild into audited table with signed modulation and per-value citations. |
| **Reliability Scorer $R$** | **Change** | Refactor from unsupervised loss to supervised BCE trained on agreement with consensus. |
| **Evidence Aggregation** | **Change** | Replace multihead cross-attention with residual form: $\text{logit}(p) = \text{logit}_{\text{prior}} + \lambda \cdot E$. |
| **Evaluation Harness** | **Change** | Shift from study-disjoint to 5×5 repeated grouped CV (herb-, drug-, pair-disjoint). |
| **12D Feature Vector** | **Demote** | Fully remove polarity columns; restrict to 7D quality and 5D design features. |
| **Convex Gate $(1-R)z_g + R z_t$** | **Demote** | Deprecate in favor of the mathematically clean residual evidence aggregation. |
| **Raw Text Embeddings** | **Demote** | Demoted strictly to Appendix as `LEAKY REFERENCE (not a valid model)`. |
| **Complex GAT + BiomedBERT** | **Demote** | Demote to exploratory comparison; primary model is interpretable LR + Aggregator. |

---

## 9. Phased Order of Work & Responsibilities

| Phase | Milestone / Deliverable | Owner | Duration |
| :--- | :--- | :--- | :--- |
| **Step 1** | Audit existing pair coverage (120 pairs); compile consensus labels table (`consensus_labels.csv`) with per-pair citations; double-annotate subset for Cohen's $\kappa$. | User | 1–2 weeks |
| **Step 2** | Rebuild mechanistic knowledge table with signed values and per-value citations; parse study design fields in Studies table. | User review, Agent assists | ~1 week (parallel) |
| **Step 3** | Implement Task A mechanistic prior, shortcut baselines, and 5×5 repeated grouped CV harness. | Agent | 3–5 days |
| **Step 4** | Implement agreement-supervised $R$ model, residual evidence aggregation ($E$), and full ablation family. | Agent | 1 week |
| **Step 5** | Implement pooled out-of-fold calibration, conformal tri-state abstention, and validation against disputed pairs. | Agent | 3–5 days |
| **Step 6** | Independent verification and reproduction run in fresh environment. | User | 2–3 days |
| **Step 7** | Rewrite publication manuscript and patent application strictly from `results/*.json`. | Both | 1–2 weeks |

---

## 10. Honest Scientific Expectations

1. **Task A (Cold-Start Mechanism):** Expect ROC-AUC in the **0.80–0.87** range, consistent with biological target overlap capabilities. The superficial 0.97 AUC observed in raw text models will disappear once outcome leakage is eliminated.
2. **Task B (Evidence Aggregation):** The learned reliability aggregator may yield modest improvements over unweighted or heuristic votes, or perform comparably. Either finding is scientifically valuable: demonstrating that literature aggregation requires accounting for study design and echo inflation is a fundamental contribution regardless of margin.
3. **Abstention on Disputed Pairs:** Selective abstention on clinically conflicting pairs represents the most impactful clinical result, demonstrating safe decision support rather than forced inaccurate predictions.

---

## 11. Manuscript & Patent Framing

### Paper Framing
1. **Systematic Shortcut Analysis:** First rigorous demonstration that literature-derived HDI benchmarks are dominated by entity memorization and outcome-word leakage.
2. **Evidence vs. Truth Decoupling:** Formal conceptual and mathematical separation between what an individual paper reports and the clinical ground truth.
3. **Supervised Reliability & Echo Discounting:** A transparent aggregation framework accounting for study quality and citation propagation.
4. **Conformal Decision Support:** Selective abstention validated on historically disputed interactions.

### Patent Framing
Focus claims on the **data processing and decision support method**, not transient performance numbers:
* Method for decoupling study findings from consensus targets using agreement-supervised reliability modeling.
* Method for citation echo-attenuated evidence pooling combined with a signed pharmacokinetic fallback prior.
* Tri-state routing architecture directing conflicting and high-uncertainty botanical interactions to specialized clinical pharmacist review.

---

## 12. Main Risks & Mitigations

1. **Risk of Circularity in Consensus Labels:**
   * *Problem:* Clinical compendia (e.g., Stockley's, Natural Medicines) might have derived their monographs from the very studies in Table 1.
   * *Mitigation:* Document primary source citations for each consensus entry; cross-reference multiple independent compendia (NaPDI, DrugBank, FDA tables).
2. **Small Sample Size & High Variance:**
   * *Problem:* Evaluating on 120–250 pairs can yield wide confidence intervals.
   * *Mitigation:* Expand unstudied negative and positive candidate pairs to reach 250–400 pairs; report 95% bootstrap confidence intervals honestly and avoid over-interpreting minor score differences.
3. **Subjectivity in Mechanistic Profile Construction:**
   * *Problem:* Hand-selected CYP/transporter modulation values could inadvertently encode knowledge of interaction outcomes.
   * *Mitigation:* Require exact primary literature citations for every enzyme potency value; perform blind verification on an audited sample by a second annotator.
