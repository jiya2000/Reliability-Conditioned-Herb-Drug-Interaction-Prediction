# Task Definitions: Decoupling Mechanistic Cold-Start from Evidence Aggregation
## Reliability-Conditioned Herb–Drug Interaction (HDI) Prediction Framework

**Document Identifier:** `DOC-TASK-DEF-2026-10-28`  
**Authors:** Research & Development Engineering Team  
**Scope:** Formal Task Definitions for Phase 1 Redesign  

---

## 1. Motivation: The Critical Separation of Evidence and Ground Truth

In standard literature-derived interaction datasets, a fundamental architectural flaw often arises: **each publication's reported finding serves simultaneously as the input evidence and the evaluation label**.

When this occurs:
1. **Text Reading Comprehension Shortcut:** Natural language models or bag-of-words classifiers achieve near-perfect performance ($0.9716$ TF-IDF AUC from `results/phase2_masked_text.json`) simply by reading the author's conclusion words (*"does not alter"*, *"did not affect"*, *"inhibition"*, *"significantly increased AUC"*), rather than predicting interactions from biological mechanisms.
2. **Untrainable Reliability:** Evidence reliability cannot be meaningfully learned, because an empirical study cannot be "unreliable" about a ground-truth label that is defined as its own conclusion.
3. **Inability to Model Genuine Contradiction:** When two publications report opposing findings (e.g., *Panax ginseng* + Digoxin), the dataset contains conflicting ground-truth rows rather than a unified entity pair with divergent evidentiary quality.

To resolve this root-cause design flaw, the framework formally decouples **individual study evidence** from **pair-level ground-truth consensus**:

- **Study-Level Evidence ($\mathcal{E}_i$):** Methodological design (species, assay type, route, preparation, dose, sample size $n$) and individual reported observation $o_i \in \{-1, 0, +1\}$.
- **Pair-Level Ground Truth ($Y_{h,d}$):** Authoritative consensus interaction label (`interaction`, `no_interaction`, `disputed`) curated from independent clinical compendia (NaPDI, Natural Medicines, Stockley's Drug Interactions, DrugBank).

---

## 2. Task A: Mechanistic Cold-Start Prediction (Zero-Literature Prior)

### 2.1 Problem Formulation
Given an unstudied botanical entity $h$ and prescription drug $d$ for which **zero empirical literature exists** ($\mathcal{E}_{h,d} = \emptyset$, literature missingness $M = 1.0$):
- **Inputs:** Strictly biological and pharmacokinetic property representations sourced exclusively from the verified curation table (`data/curation/mechanistic_features.csv`). Legacy 16-D profile vectors from `src/features/biological_features.py` are strictly exempt only for the labeled legacy baseline comparison row:
  1. Pharmacokinetic target overlap features: $\mathbf{x}_{\text{overlap}} = [|h_j \cdot d_j|]_{j \in \text{CYPs, transporters}} \in \mathbb{R}^6$.
  2. Pharmacokinetic site-of-action reachability: $\rho_{\text{reach}} = \min(1.0, [I]_{u,\max} / \text{potency})$.
  3. Fraction of clearance mediated: $f_{m,\text{CYP}}$.
  - **Forbidden Inputs:** Zero study text, zero citation lineage features, zero polarity counts, zero literature abstracts.
- **Output:** Calibrated prior interaction probability:
  $$\hat{p}_{\text{prior}} = \sigma(\text{logit}_{\text{prior}}(\mathbf{x}_{\text{mechanistic}})) \in [0, 1]$$
- **Target:** Pair-level consensus label $Y_{h,d} \in \{0, 1\}$.

### 2.2 Clinical & Regulatory Purpose
Task A solves the de novo screening problem: assessing theoretical interaction risk for novel dietary supplements or investigational drugs before clinical trials are conducted.

### 2.3 Evaluation Protocol
- Strict **`herb-disjoint`** (evaluating completely unseen botanicals), **`drug-disjoint`** (evaluating completely unseen drugs), and **`pair-disjoint`** cross-validation.
- Benchmarked against simple baseline models (L2-regularized Logistic Regression, HistGradientBoosting, small regularized GAT).

---

## 3. Pair Study Statistics & Feasibility Analysis

- Total unique herb–drug pairs in benchmark: **120**
- Pairs with 2 or more studies: **15** (out of 120 total pairs)
- Study count distribution: 105 pairs with 1 study, 11 pairs with 2 studies, 2 pairs with 3 studies, 2 pairs with 4 studies

Given that only **15 out of 120** pairs have 2 or more studies (well below the required threshold of 20), **Task B is marked "not testable on this panel."** The limited number of pairs with multiple study citations means that reliability conditioning on study agreement cannot be meaningfully evaluated on this 141-record panel. Task B requires a substantially larger benchmark with multi-study evidence per pair.

---

## 4. Task B: Supervised Evidence Aggregation & Reliability Conditioning

### 4.1 Problem Formulation
Given a pair $(h, d)$ with a collection of $k \ge 1$ empirical study publications $\mathcal{E}_{h,d} = \{(d_i, o_i, \text{lineage}_i)\}_{i=1}^k$:
- **Study Inputs ($d_i$):** Methodological design parameters ONLY:
  - Species ($w_{\text{species}} \in [0.20, 1.00]$: human in vivo, rodent in vivo, cell line in vitro)
  - Assay type ($w_{\text{assay}}$: clinical RCT, pharmacokinetic cohort, case report, in vitro CYP inhibition)
  - Route (oral, intravenous, topical)
  - Preparation (standardized extract, crude powder, isolated constituent)
  - Exposure dose and sample size $n$
- **Citation Lineage ($\text{lineage}_i$):** Citation network depth, derivative echo ratio ($\text{echo\_ratio}$), source publication tier, and independent replication count.
- **Study Observation ($o_i$):** Directional finding:
  $$o_i = \begin{cases} +1 & \text{if study observed significant interaction} \\ -1 & \text{if study observed no significant interaction} \\ 0 & \text{if study was inconclusive / equivocal} \end{cases}$$
- **Supervised Reliability Scoring ($R_i$):**
  Each study's reliability $R_i \in [0, 1]$ is supervised by whether study $i$'s finding $o_i$ agrees with the pair consensus $Y_{h,d}$. Agreement is defined as $o_i == +1 \land Y_{h,d} == 1$ or $o_i == -1 \land Y_{h,d} == 0$, with equivocal / inconclusive observations ($o_i == 0$) excluded:
  $$\text{target}(R_i) = \begin{cases} 1 & \text{if } (o_i == +1 \land Y_{h,d} == 1) \lor (o_i == -1 \land Y_{h,d} == 0) \\ 0 & \text{if } (o_i == +1 \land Y_{h,d} == 0) \lor (o_i == -1 \land Y_{h,d} == 1) \\ \text{excluded} & \text{if } o_i == 0 \end{cases}$$
  This operationalizes the core scientific thesis: human clinical RCTs will consistently achieve high reliability ($R \to 1.0$), whereas irreproducible cell-culture assays or derivative review echo chambers will receive low reliability ($R \to 0.0$).

### 4.2 Residual Aggregation Architecture
The global prediction combines the cold-start mechanistic prior with the reliability-weighted evidentiary consensus:
$$\text{logit}_{\text{fused}} = \text{logit}_{\text{prior}}(\mathbf{x}_{\text{mechanistic}}) + \sum_{i=1}^k R_i \cdot \delta(o_i)$$
where $\delta(o_i) \in \mathbb{R}$ represents the evidentiary adjustment contribution.

#### Mathematical Invariance Properties:
1. **Zero-Literature Fallback Guarantee:** When $k = 0$ (no literature) or all studies have missingness $M=1.0$, the evidentiary summation $\sum R_i \delta_i = 0$. The model strictly simplifies to:
   $$\text{logit}_{\text{fused}} \equiv \text{logit}_{\text{prior}}$$
   The architecture cannot perform worse than the cold-start pharmacokinetic prior.
2. **Echo Chamber Attenuation:** If 10 derivative review papers cite a single in vitro observation, their effective reliability is discounted by $\text{echo\_ratio}$, preventing redundant publications from overwhelming a single dissenting clinical trial.

---

## 5. Conformal Tri-State Decision Routing

For both Task A and Task B, predictions are passed through split-conformal prediction sets:
$$\mathcal{C}(X) \subseteq \{0, 1\}, \quad P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$$

Decisions are mapped into three actionable clinical routes:
1. $\mathcal{C}(X) = \{1\}$: **Interaction Present** $\to$ CDS advisory alert for dose adjustment or therapeutic alternative.
2. $\mathcal{C}(X) = \{0\}$: **No Interaction Observed** $\to$ Safe co-administration clearance.
3. $\mathcal{C}(X) = \{0, 1\}$ or $\emptyset$: **Abstain / Indeterminate** $\to$ High-priority routing to clinical pharmacist specialist for manual case review.

---

## 6. Summary Matrix of Task Differences

| Dimension | Task A: Mechanistic Cold-Start | Task B: Supervised Evidence Aggregation |
| :--- | :--- | :--- |
| **Available Literature** | Zero literature ($k=0, M=1.0$) | $k \ge 1$ empirical study publications |
| **Input Features** | 6D CYP/transporter overlap, $f_m$, $\rho_{\text{reach}}$ | Study design (species, assay, dose) + Citation lineage |
| **Text Usage** | **Strictly Forbidden (Zero text)** | Design parsing ONLY (Outcome language stripped) |
| **Reliability $R$** | Inactive ($R \equiv 0.0$) | Active ($R_i \in [0, 1]$ learned from consensus agreement) |
| **Clinical Use Case** | De novo drug/supplement development screening | Evidence-based clinical guidelines & pharmacist decision support |
| **Primary Evaluation** | `herb-disjoint`, `drug-disjoint`, `pair-disjoint` CV | Grouped citation cluster CV against consensus labels |
