# PHASE 0 AUDIT: RELIABILITY-CONDITIONED HERB–DRUG INTERACTION (HDI) PREDICTION
**Author / Auditor:** Antigravity Agentic Audit Suite  
**Date:** 2026-10-27  
**Scope:** `src/data`, `src/features`, `src/models`, `src/training`, `src/explainability`, `benchmarks/`, `data/benchmark/`

---

## 1. How is the Interaction Label Derived?

In the current repository, the interaction label is derived from `data/benchmark/verified_botanical_hdi.csv`.

### Field Tracing
1. **Source Column:** `outcome` in `verified_botanical_hdi.csv`.
2. **Values Present:**
   - `interaction_present`: 75 records (53.2%)
   - `no_interaction_observed`: 62 records (44.0%)
   - `conflicting`: 4 records (2.8%)
3. **Downstream Label Mapping (`benchmarks/run_verified_hdi_benchmark.py:175-182` and `scripts/verify_app_functionality.py:67-73`):**
   ```python
   out = str(row["outcome"]).lower()
   if out in {"interaction_present", "positive"}:
       y = 1.0
   elif out in {"no_interaction_observed", "negative"}:
       y = 0.0
   else:
       y = 0.5  # Conflicting / uncertain
   ```
4. **Binary Evaluation Binarization (`src/training/evaluation.py:326`):**
   ```python
   y_true_arr = np.asarray(y_true, dtype=int).flatten()
   ```
   - When `y = 0.5` is cast to integer (`int(0.5)`), it evaluates to `0`. Consequently, all 4 conflicting records are treated as negative (class 0) during ROC-AUC, PR-AUC, and F1 calculation.
5. **Auxiliary / Corroborating Columns in CSV:**
   - `raw_label`: e.g., `cyclosporine_clearance_increase`, `auc_decrease`, `no_change_inr`, `no_effect`.
   - `effect_direction`: `no_effect` (62), `decreased_exposure` (30), `increased_exposure` (26), `increased_toxicity` (13), `decreased_effect` (6), `contradictory` (4).
   - `clinical_significance`: `major`, `moderate`, `minor`, `none`, `uncertain`.

---

## 2. Breakdown of the 12D Lineage Features: Label-Adjacent vs. Quality-Only

In `src/data/evidence_lineage.py`, `PairEvidenceSummary.to_full_metadata_vector()` generates a 12-dimensional vector. Below is the rigorous audit of each dimension:

| Dimension Index | Feature Identifier | Mathematical Definition | Derivation Source | Classification | Leakage / Risk Analysis |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | `S_pos` | $\sum_{k} S(\mathcal{C}_k^+)$ | Summed over records where `outcome in {"interaction_present", "positive"}` | **LABEL-ADJACENT** | **Severe Leakage Risk:** Encodes test partition label directly if computed over test records. |
| **1** | `S_neg` | $\sum_{k} S(\mathcal{C}_k^-)$ | Summed over records where `outcome in {"no_interaction_observed", "negative"}` | **LABEL-ADJACENT** | **Severe Leakage Risk:** Encodes test partition label directly if computed over test records. |
| **2** | `echo_ratio` | $\max(0, 1 - \frac{\text{clusters}}{\text{records}})$ | Citation graph topology ($N_{\text{clusters}} / N_{\text{records}}$) | **QUALITY-ONLY** | Valid bibliometric signal independent of positive/negative claim polarity. |
| **3** | `q_mean` | $\frac{1}{N}\sum q_i \cdot \text{conf}_i$ | Study type weight (RCT=0.95, clinical=0.90, in vitro=0.20) | **QUALITY-ONLY** | Valid study quality metric independent of study outcome. |
| **4** | $W_{\text{context}}$ | $\frac{1}{N}\sum (w_{\text{sp}} \cdot w_{\text{pr}} \cdot w_{\text{rt}})$ | Species, preparation, and route metadata | **QUALITY-ONLY** | Context compatibility score independent of interaction outcome. |
| **5** | $C_{h,d}$ | $\frac{2\min(S_{\text{pos}}, S_{\text{neg}})}{S_{\text{pos}} + S_{\text{neg}} + \epsilon}$ | Ratio of positive and negative echo-discounted strengths | **LABEL-ADJACENT** | Derived directly from $S_{\text{pos}}$ and $S_{\text{neg}}$. When $S_{\text{pos}} > 0$ and $S_{\text{neg}} = 0$, $C_{h,d} \equiv 0$. |
| **6** | `margin` | $S_{\text{pos}} - S_{\text{neg}}$ | Difference of echo-attenuated strengths | **LABEL-ADJACENT** | **Severe Leakage Risk:** Exactly tracks interaction sign ($+1$ for positive, $-1$ for negative). |
| **7** | `missingness` | $\frac{\text{missing fields}}{4N}$ | Fraction of missing clinical fields (dose, route, prep, species) | **QUALITY-ONLY** | Valid data-completeness indicator independent of label. |
| **8** | $N_{\text{raw\_pos}}$ | $\text{len}(\text{pos\_recs})$ | Count of records with `outcome == "interaction_present"` | **LABEL-ADJACENT** | **Severe Leakage Risk:** Direct count of ground-truth positive labels. |
| **9** | $N_{\text{raw\_neg}}$ | $\text{len}(\text{neg\_recs})$ | Count of records with `outcome == "no_interaction_observed"` | **LABEL-ADJACENT** | **Severe Leakage Risk:** Direct count of ground-truth negative labels. |
| **10** | $\rho_{\text{reach}}$ | $\min(1, [I]_{u,\max}/\text{IC}_{50})$ | Pharmacokinetic reachability ratio from FDA guidance | **QUALITY-ONLY** | Valid physicochemical/pharmacokinetic parameter. |
| **11** | $f_{m,\text{CYP}}$ | Fractional metabolic clearance | Substrate elimination fraction via target CYP enzyme | **QUALITY-ONLY** | Drug-intrinsic pharmacokinetic property. |

### Critical Takeaway for Phase 1:
A model claiming to predict whether an herb and drug interact cannot be fed $S_{\text{pos}}$, $S_{\text{neg}}$, `margin`, $N_{\text{raw\_pos}}$, and $N_{\text{raw\_neg}}$ at test time if those features were aggregated from the test records. If evaluated on a held-out test study, those features provide the ground-truth label directly to the model.

---

## 3. Train–Test Entity & Pair Overlap Under `split_study_disjoint`

Using `seed=42`, `train_ratio=0.6`, `val_ratio=0.2` on the 141 records:
- **Partition Record Sizes:** 90 Train records, 25 Validation records, 26 Test records.
- **Independent Study Clusters:** 73 Train clusters, 25 Validation clusters, 25 Test clusters (123 total).

### Quantitative Overlap Audit:
1. **(Herb, Drug) Pair Overlap:**
   - Distinct Train Pairs: **72**
   - Distinct Val Pairs: **25**
   - Distinct Test Pairs: **24**
   - **Train & Test Pair Intersection:** **0 pairs (0.0% overlap)**
   - **Train & Val Pair Intersection:** **0 pairs (0.0% overlap)**
   - **Val & Test Pair Intersection:** **1 pair** (`('HERB_GINSENG', 'DRUG_DIGOXIN')`)
2. **Herb Entity Overlap:**
   - Distinct Train Herbs: **18**
   - Distinct Val Herbs: **14**
   - Distinct Test Herbs: **13**
   - **Train & Test Herb Intersection:** **13 herbs (100.0% of test herbs exist in train)**
   - **Unseen Herbs in Test:** **0** (All test herbs have appeared in training data with different drugs).
3. **Drug Entity Overlap:**
   - Distinct Train Drugs: **31**
   - Distinct Val Drugs: **17**
   - Distinct Test Drugs: **15**
   - **Train & Test Drug Intersection:** **10 drugs (66.7% of test drugs exist in train)**
   - **Unseen Drugs in Test:** **5 drugs** (`DRUG_LEVODOPA`, `DRUG_CAFFEINE`, `DRUG_PACLITAXEL`, `DRUG_DOCETAXEL`, `DRUG_ROSUVASTATIN`).

### Evaluation Finding:
`split_study_disjoint` is strictly *study-disjoint* and *almost pair-disjoint* (with 0 train/test pair overlap), but it is **NOT herb-disjoint** (all herbs in test were seen in train) and **NOT drug-disjoint** (67% of drugs were seen in train). True zero-shot botanical generalization requires explicit `herb_disjoint` and `drug_disjoint` split modes (to be implemented in Phase 5).

---

## 4. Origin & Contradiction of the $R = 0.95$ Fallback for Missing Literature

### Where it Comes From:
In `scripts/verify_app_functionality.py` (and previously in `app/interactive_app.py`), the test output for Case 4 (Saw Palmetto + Acetaminophen, an unstudied de novo pair) reported:
```text
CASE 4 (De Novo Unstudied: Saw Palmetto + Acetaminophen):
   Decision State    : NO INTERACTION
   Calibrated Prob   : 27.58%
   Reliability (R)   : 0.95
   Has Literature    : False
```

### Why Did $R$ Evaluate to 0.95?
1. In `verify_app_functionality.py:140-145`, when no literature exists for a pair:
   ```python
   text_tensor = torch.zeros((1, 16), dtype=torch.float32)
   meta_vector = np.zeros(12, dtype=np.float32)
   meta_vector[7] = 1.0  # missingness = 1.0
   meta_vector[4] = 1.0  # context_mult = 1.0
   ```
2. In `src/models/reliability_scorer.py`, `ReliabilityScorer` is an MLP terminating in a Sigmoid:
   $$\hat{R} = \sigma(\text{MLP}(\mathbf{m}))$$
3. Because all 141 records in `verified_botanical_hdi.csv` are published literature records from PubMed, **no training instance had `missingness = 1.0` and `S_pos = S_neg = 0`**.
4. At inference time, the unconstrained MLP evaluated an out-of-distribution input where `missingness = 1.0` and produced an uncalibrated latent output that passed through the sigmoid as $\hat{R} \approx 0.95$.

### Why This Contradicts the Fundamental Mathematical Design:
The paper and patent design explicitly formulate multimodal fusion as:
$$\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$$
- **Theoretical Mandate:** When scientific literature is missing (`missingness = 1.0`), literature reliability must satisfy $R \to 0$ (or $R \le 0.05$). Under $R = 0$, $\mathbf{z}_{\text{fused}}$ mathematically collapses to:
  $$\mathbf{z}_{\text{fused}} = (1 - 0)\mathbf{z}_{\text{graph}} + 0 \cdot \mathbf{z}_{\text{contextual}} = \mathbf{z}_{\text{graph}}$$
  defaulting 100% of the prediction weight to the biological GAT mechanism prior.
- **The Contradiction:** With $R = 0.95$, the model assigned 95% of its weight to an empty zeroed text embedding $\mathbf{z}_{\text{contextual}} = \mathbf{0}$, and only 5% of its weight to the biological graph prior $\mathbf{z}_{\text{graph}}$.
- **Required Fix (Phase 2):** Explicitly enforce that when `missingness == 1.0` (or $S_{\text{pos}} = S_{\text{neg}} = 0$ with no literature), the reliability module enforces $R \le 0.05$ (e.g., via $R = R_{\text{raw}} \cdot (1.0 - \text{missingness})$ or architectural masking).

---

## 5. How the Citation DAG is Built, Data Sources Used, and Failure Modes

### Construction Process:
1. **Source Data:** The `cited_study_ids` column in `data/benchmark/verified_botanical_hdi.csv`.
2. **Graph Model (`src/data/evidence_lineage.py:190-240`):**
   - Nodes represent `study_id` (e.g., `STUDY_RUSCHITZKA_1999`, `STUDY_BREIDENBACH_2000`).
   - Directed edges $(u, v)$ represent citation links: paper $u$ cites paper $v$.
   - Clusters are derived by converting the graph to an undirected view (`lineage_graph.to_undirected()`) and computing `nx.connected_components()`.
   - The primary index study is assigned as the earliest study or the root with in-degree zero.
   - Derivative papers in the component receive echo discount $\alpha = 0.15$.

### Failure Modes of the Current Citation DAG:
1. **Manual Annotation Sparsity:** In the 141-record CSV, only 5 records have explicit `cited_study_ids` populated (e.g., Breidenbach 2000 citing Ruschitzka 1999). The remaining 136 records have empty citation lists `[]`. Consequently, derivative reviews that do not have manual annotation are treated as independent roots.
2. **Disconnected Disjoint Identifiers:** If two independent teams cite the same seminal clinical trial but use slightly different identifier strings (e.g., `PUBMED_10534248` vs `STUDY_RUSCHITZKA_1999`), the graph fails to create an edge and treats them as independent.
3. **Loss of Citation Directionality in Undirected Clustering:** Converting the DAG to an undirected graph to find connected components discards the arrow of time and citation direction. If paper A cites B and paper C cites B, all three are merged into one cluster even if A and C are independent clinical trials with different outcomes.
4. **No Automated Bibliographic Ingestion at Runtime:** The system relies entirely on static CSV columns and has no active API fallback (e.g., PubMed E-utilities or OpenAlex) to automatically discover citations for newly queried papers.
5. **Polarity Blindness (Refutation as Echo):** If Study B cites Study A explicitly to *refute* its findings (e.g., "Study A reported toxicity; however, in our rigorous RCT we proved it was an artifact"), the DAG treats Study B as an echo of Study A, discounting Study B by 85% ($\alpha = 0.15$) rather than recognizing it as a refutation.

---

## 6. Audit Conclusion & Phase 0 Sign-Off
Phase 0 establishes that the fundamental multimodal GAT + calibration architecture is structurally sound, but the evaluation pipeline must be systematically hardened to:
1. Prevent label leakage by decoupling evidentiary quality/provenance from claim polarity.
2. Formally enforce $R \to 0$ when literature is absent.
3. Expand split modes beyond `study_disjoint` to `pair_disjoint`, `herb_disjoint`, and `drug_disjoint`.
