# Provenance and Leakage Risk Audit of Mechanistic Features
## Reliability-Conditioned Herb–Drug Interaction (HDI) Prediction Framework

**Document Identifier:** `REP-PROVENANCE-AUDIT-2026-10-27-REV2`  
**Authors:** Internal Engineering & Audit Team (Note: Not an independent third-party audit; written by the same team maintaining the repository)  
**Scope:** Phase 0b Mechanistic Feature Provenance, Selection Adjacency, & Leakage Assessment  
**Reference Codebase:** `src/features/biological_features.py`, `benchmarks/run_verified_hdi_benchmark.py`, `src/training/evaluation.py`  
**Dataset:** `data/benchmark/verified_botanical_hdi.csv` (141 records across 18 herbs and 40 drugs)  
**Per-Value Provenance Table:** [reports/profile_provenance.csv](reports/profile_provenance.csv) (111 parameter entries)

---

## 1. Executive Summary

This audit investigates the provenance, mathematical formulation, parameter-level provenance, and potential label leakage of the biological target features in `BiologicalProfileExtractor`. Specifically:
1. **Herb Profiles:** Botanical modulatory vectors (`HERB_BIOLOGICAL_PROFILES`) are reclassified as **LABEL-ADJACENT (pending per-value citation)**. The values reflect qualitative literature knowledge, but were committed to the repository after the benchmark dataset labels existed and encode outcome-level classifications (such as designating specific herbs as negative controls).
2. **Drug Profiles:** Prescription drug clearance pathways (`DRUG_BIOLOGICAL_PROFILES`) are reclassified as **pathway values independent of labels, but drug selection is label-adjacent (the 40 drugs were chosen because they appear in herb-drug studies)**.
3. **Disjoint Generalization:** The overlap signal survives herb-disjoint splits on this 18-herb, 141-record panel; it does not establish generalization to arbitrary botanicals.
4. **Partition Integrity:** The claim that `herb_onehot = 0.500` under herb-disjoint splits proves "zero partition leakage" has been retracted; zero-AUC for unseen one-hot features is mathematically guaranteed for unseen categories. Partition disjointness is now strictly verified via explicit code assertions in `LeakageSafeSplitter.grouped_kfold_split` and dedicated unit tests.

---

## 2. Feature-by-Feature Code Extraction & Mathematical Definition

### 2.1 Canonical Biological Targets (`CANONICAL_TARGETS`)
- **Location:** `src/features/biological_features.py:21-38`
- **Definition:** A 16-dimensional basis vector representing the primary human pharmacokinetic enzymes, transporters, and pharmacodynamic receptors:
  - Phase I Metabolic Enzymes: `CYP3A4` (idx 0), `CYP2C9` (idx 1), `CYP2D6` (idx 2), `CYP1A2` (idx 3), `CYP2C19` (idx 4), `CYP2E1` (idx 5)
  - Transporters: `ABCB1` (P-glycoprotein, idx 6), `SLCO1A2` (OATP1A2, idx 7), `SLCO1B1` (OATP1B1, idx 8)
  - Steroid & Nuclear Receptors: `HSD11B2` (idx 9), `NR1I2` (PXR, idx 15)
  - Pharmacodynamic Receptors: `VKORC1` (idx 10), `PTGS1` (idx 11), `PTGS2` (idx 12), `GABRA1` (idx 13), `PTAFR` (idx 14)
- **Classification:** **Canonical pharmacological definitions.** Targets correspond to standard clinical interaction testing pathways recommended by FDA and EMA guidance.

---

### 2.2 Botanical Biological Profiles (`HERB_BIOLOGICAL_PROFILES`)
- **Location:** `src/features/biological_features.py:47-143`
- **Classification:** **LABEL-ADJACENT (pending per-value citation)**
- **Implementation:** Hardcoded dictionary mapping 18 botanical entity identifiers (`HERB_HYPERICUM`, `HERB_GINKGO`, etc.) to sparse continuous target modulatory values:
  - Values represent modulatory direction and potency:
    - $+1.0$: Strong in vivo induction/activation (e.g., Hypericum on `NR1I2`, `CYP3A4`, `ABCB1`)
    - $-1.0$: Strong mechanism-based or irreversible inhibition (e.g., Goldenseal on `CYP3A4`, Grapefruit on `CYP3A4`, Licorice on `HSD11B2`)
    - $-0.5$ to $-0.8$: Moderate inhibition (e.g., Curcuma on `CYP3A4` $-0.7$, `ABCB1` $-0.8$; Kava on `CYP3A4` $-0.9$, `CYP2C9` $-0.8$)
    - $0.0$ / near $0.0$: Explicitly designated "neutral / negative control" herbs (e.g., Valerian on `CYP3A4` $0.0$, `ABCB1` $0.0$; Saw Palmetto on `CYP3A4` $0.0$, `ABCB1` $0.0$; Black Cohosh on `CYP3A4` $0.0$, `ABCB1` $0.0$; Milk Thistle on `CYP3A4` $0.05$, `ABCB1` $0.05$).
- **Audit Assessment:**
  - *Label-Adjacency and Outcome Knowledge:* Designating herbs as "negative controls" inside a profile dictionary is a label-level concept. In biomedical pharmacology, the characterization of St. John's Wort as a potent CYP3A4/P-gp inducer arose historically from clinical interaction discoveries with cyclosporine and indinavir. Setting CYP3A4 to 0.0 for Valerian or Saw Palmetto was done knowing their non-interacting clinical status.
  - *Commit Chronology:* Git log and blame analysis reveals `data/benchmark/verified_botanical_hdi.csv` was added in commit `8ebd18c` (2026-10-14 11:32:49 +0530), whereas `src/features/biological_features.py` was introduced in commit `79b1ada` (2026-10-14 11:57:12 +0530). The profiles were therefore committed after the benchmark labels existed.
  - *Per-Value Citation Status:* All 111 profile entries lack formal DOIs or PMIDs in code comments; source details and retrieval dates remain **UNVERIFIED** (see Section 6).

---

### 2.3 Prescription Drug Biological Profiles (`DRUG_BIOLOGICAL_PROFILES`)
- **Location:** `src/features/biological_features.py:148-287`
- **Classification:** **Pathway values independent of labels, but drug selection is label-adjacent (the 40 drugs were chosen because they appear in herb-drug studies)**
- **Implementation:** Hardcoded dictionary mapping 40 prescription drug identifiers (`DRUG_CYCLOSPORINE`, `DRUG_MIDAZOLAM`, `DRUG_WARFARIN`, etc.) to continuous dependency values:
  - Values represent clearance pathway dependency or target sensitivity (e.g., Midazolam $\to$ `CYP3A4`: $1.0$; Digoxin $\to$ `ABCB1`: $1.0$; Warfarin $\to$ `CYP2C9`: $1.0$).
- **Audit Assessment:**
  - The pathway values themselves (e.g., Midazolam cleared by CYP3A4) are standard clinical pharmacokinetics independent of botanicals.
  - However, the **selection of these 40 specific drugs** is label-adjacent: these drugs were selected precisely because they are index substrates in published herb-drug clinical trials. In a broader pharmacopeia, the prior distribution of drug clearance pathways would differ substantially.

---

### 2.4 Pair Feature Construction & Code Hygiene
- **Location:** `src/features/biological_features.py:317-327`, `benchmarks/run_verified_hdi_benchmark.py:70-79`
- **Hadamard Overlap:**
  $$\text{interaction\_vec} = h\_vec \odot d\_vec, \quad \text{overlap\_total} = \sum_{j=1}^{16} |h_j \cdot d_j|$$
- **Code Hygiene Remediation:**
  In `benchmarks/run_verified_hdi_benchmark.py:70-74`, index lookups previously utilized fallback default indices:
  `TARGET_TO_IDX.get("ABCB1", 1)` (canonical index is 6, fallback was 1) and `TARGET_TO_IDX.get("CYP2C9", 2)` (canonical index is 1, fallback was 2).
  While the keys existed in `TARGET_TO_IDX`, this pattern presented risk of silent data corruption. All lookups have been replaced with strict dictionary indexing `TARGET_TO_IDX[target]`, verified by unit tests asserting that missing keys raise `KeyError`.

---

## 3. Disjoint Generalization Analysis

To evaluate generalization, 5-fold grouped cross-validation was evaluated across three partitioning regimes:

| Feature Representation | `pair-disjoint` CV AUC | `drug-disjoint` CV AUC | `herb-disjoint` CV AUC | Empirical Diagnostic |
| :--- | :---: | :---: | :---: | :--- |
| **6D Overlap Only** | **0.868 ± 0.022** | **0.863 ± 0.090** | **0.840 ± 0.098** | Overlap signal survives herb-disjoint splits on this 18-herb, 141-record panel; it does not establish generalization to arbitrary botanicals. |
| **Combined 13D** | **0.883 ± 0.032** | **0.880 ± 0.081** | **0.862 ± 0.092** | Quality features add +0.01 to +0.02 across split types. |
| **7D Quality Lineage** | 0.652 ± 0.076 | 0.642 ± 0.117 | 0.658 ± 0.019 | Modest baseline without biological targets. |
| **Herb One-Hot Only** | 0.800 ± 0.125 | 0.792 ± 0.058 | **0.500 ± 0.000** | Guaranteed for unseen categories; does not verify zero leakage elsewhere. |

### Evaluation Correction:
1. **Retraction of Proof Claim:** The observation that `herb_onehot` achieves exactly 0.500 under herb-disjoint splits is a trivial mathematical consequence of test herbs having zero active one-hot indicator bits at inference time. It does **not** prove zero leakage of the underlying pipeline.
2. **Explicit Verification of Partition Disjointness:** Partition disjointness is now explicitly verified in code:
   `LeakageSafeSplitter.grouped_kfold_split` enforces `assert len(set(train_df[entity]) & set(test_df[entity])) == 0` for all folds, backed by automated unit tests in `tests/test_audit_and_leakage_checks.py`.

---

## 4. Benchmark Panel Selection & Outcome Distribution

### 4.1 How the 141 Records Were Selected
Audit of `data/benchmark/ANNOTATION_PROTOCOL.md` and repository curation scripts reveals:
1. **Deliberate Negative Controls:** Negative records ($Y=0$, $N=62$) were not sampled randomly or under identical clinical study criteria as positives. Rather, specific clinical trials demonstrating absence of alteration in drug AUC, clearance, or pharmacodynamics were deliberately identified and curated as true negative controls.
2. **Four Exclusive Negative Control Herbs:** Four botanicals in the panel—Black Cohosh ($N=5$, 0 pos / 5 neg), Milk Thistle ($N=9$, 0 pos / 9 neg), Saw Palmetto ($N=5$, 0 pos / 5 neg), and Valerian ($N=6$, 0 pos / 6 neg)—have 0 positive interaction records across all 25 pairs.
3. **Outcome Balance:** Across the 141 records:
   - `interaction_present` ($Y=1$): 75 records (53.2%)
   - `no_interaction_observed` ($Y=0$): 62 records (44.0%)
   - `conflicting`: 4 records (2.8%)

### 4.2 Outcome Breakdown by Herb Entity (18 Herbs)

| Herb ID | Conflicting | Interaction Present ($Y=1$) | No Interaction Observed ($Y=0$) | Total Records | Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `HERB_HYPERICUM` | 0 | 20 | 1 | 21 | 95.2% |
| `HERB_GINKGO` | 0 | 8 | 7 | 15 | 53.3% |
| `HERB_GREENTEA` | 0 | 7 | 3 | 10 | 70.0% |
| `HERB_GINSENG` | 2 | 4 | 4 | 10 | 40.0% |
| `HERB_GARLIC` | 0 | 4 | 5 | 9 | 44.4% |
| `HERB_MILKTHISTLE` | 0 | 0 | 9 | 9 | **0.0%** (Control) |
| `HERB_CURCUMA` | 0 | 6 | 2 | 8 | 75.0% |
| `HERB_ECHINACEA` | 2 | 1 | 5 | 8 | 12.5% |
| `HERB_GOLDENSEAL` | 0 | 6 | 1 | 7 | 85.7% |
| `HERB_LICORICE` | 0 | 5 | 2 | 7 | 71.4% |
| `HERB_GRAPEFRUIT` | 0 | 4 | 2 | 6 | 66.7% |
| `HERB_VALERIAN` | 0 | 0 | 6 | 6 | **0.0%** (Control) |
| `HERB_BLACKCOHOSH` | 0 | 0 | 5 | 5 | **0.0%** (Control) |
| `HERB_KAVA` | 0 | 5 | 0 | 5 | 100.0% |
| `HERB_SAWPALMETTO` | 0 | 0 | 5 | 5 | **0.0%** (Control) |
| `HERB_CRANBERRY` | 0 | 1 | 3 | 4 | 25.0% |
| `HERB_CHAMOMILE` | 0 | 2 | 1 | 3 | 66.7% |
| `HERB_PEPPERMINT` | 0 | 2 | 1 | 3 | 66.7% |
| **Total** | **4** | **75** | **62** | **141** | **53.2%** |

### 4.3 Outcome Breakdown by Drug Entity (Top 15 Drugs)

| Drug ID | Conflicting | Interaction Present ($Y=1$) | No Interaction Observed ($Y=0$) | Total Records | Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DRUG_WARFARIN` | 0 | 10 | 11 | 21 | 47.6% |
| `DRUG_MIDAZOLAM` | 2 | 4 | 9 | 15 | 26.7% |
| `DRUG_DIGOXIN` | 1 | 3 | 11 | 15 | 20.0% |
| `DRUG_DEXTROMETHORPHAN` | 0 | 1 | 11 | 12 | 8.3% |
| `DRUG_CYCLOSPORINE` | 0 | 7 | 0 | 7 | 100.0% |
| `DRUG_INDINAVIR` | 0 | 3 | 2 | 5 | 60.0% |
| `DRUG_THEOPHYLLINE` | 0 | 2 | 3 | 5 | 40.0% |
| `DRUG_ALPRAZOLAM` | 0 | 3 | 1 | 4 | 75.0% |
| `DRUG_ASPIRIN` | 0 | 3 | 1 | 4 | 75.0% |
| `DRUG_TACROLIMUS` | 0 | 3 | 1 | 4 | 75.0% |
| `DRUG_CHLORZOXAZONE` | 0 | 2 | 1 | 3 | 66.7% |
| `DRUG_NIFEDIPINE` | 0 | 3 | 0 | 3 | 100.0% |
| `DRUG_ROSUVASTATIN` | 0 | 1 | 2 | 3 | 33.3% |
| `DRUG_SIMVASTATIN` | 0 | 3 | 0 | 3 | 100.0% |
| `DRUG_DOCETAXEL` | 0 | 0 | 2 | 2 | 0.0% |

*(Remaining 25 drugs account for 1 to 2 records each; see full data in `data/benchmark/verified_botanical_hdi.csv`)*

---

## 5. Limitations

The following limitations must be recognized regarding mechanistic features in this benchmark:
1. **Negative Controls are a Label-Level Concept in Profile Dictionaries:** The "neutral / negative control" herbs (Valerian, Saw Palmetto, Black Cohosh, Milk Thistle) have all major CYP enzymes set to 0.0 in `HERB_BIOLOGICAL_PROFILES`. These values were set knowing the expected clinical outcome (that these botanicals do not cause pharmacokinetic interactions). This constitutes a label-level concept embedded inside an input profile dictionary.
2. **Curation Shortcut in `overlap_total`:** Because negative control herbs have target values of 0.0, their `overlap_total` is near 0.0 for every drug. Consequently, a non-negligible component of the model's predictive power is simply detecting whether an herb was curated with non-zero modulatory values ("is this herb a known modulator?").
3. **Lack of Independent Audit:** This audit was performed by the same engineering and research team that developed the model and codebase. It is not an independent third-party audit and must be interpreted with appropriate skepticism.
4. **Statistical Significance and Standard Deviations:** In cross-validation on this 141-record panel, fold metric standard deviations range from **0.08 to 0.12**. Differences of **0.01 to 0.02** between feature configurations or models (e.g., 0.874 vs 0.883) are within experimental noise and cannot be claimed as statistically meaningful improvements without rigorous hypothesis testing or bootstrap confidence intervals.

---

## 6. Per-Value Parameter Provenance Table

A comprehensive parameter audit has been generated in [reports/profile_provenance.csv](reports/profile_provenance.csv). Summary statistics:
- **Total Profile Entries:** 111 (51 herb targets across 18 botanicals; 60 drug targets across 40 drugs).
- **Source Citation:** 100% (111/111) are marked as **`UNVERIFIED`** because the codebase contains no formal DOI, PMID, or primary literature references for individual profile values.
- **Retrieval Date & Curator:** 100% (111/111) are marked as **`UNVERIFIED`**.
- **Written Before Seeing Labels:** 100% (111/111) are marked as **`no`** based on git commit history (`data/benchmark/verified_botanical_hdi.csv` committed in `8ebd18c` at 11:32:49, prior to `src/features/biological_features.py` in `79b1ada` at 11:57:12).

---

## 7. Leakage Risk Summary Table (Revised)

| Component | Source / File | Leakage Classification | Status & Operational Directive |
| :--- | :--- | :---: | :--- |
| `original_text` (Abstracts) | Benchmark CSV | 🚨 **CRITICAL SHORTCUT (0.971 AUC)** | Contains explicit outcome and conclusion phrases (`"does not alter"`, `"inhibition"`, `"increased AUC"`). Outcome terms and study findings must be masked. |
| `HERB_BIOLOGICAL_PROFILES` | `src/features/biological_features.py` | ⚠️ **LABEL-ADJACENT (pending per-value citation)** | Hardcoded profile values encode negative control classifications; git history confirms profiles added after benchmark labels existed. |
| `DRUG_BIOLOGICAL_PROFILES` | `src/features/biological_features.py` | ⚠️ **PATHWAY VALUES INDEPENDENT, DRUG SELECTION ADJACENT** | Established Phase I clearance targets, but 40 drugs chosen specifically because they appear in botanical clinical studies. |
| `interaction_vec` ($h \odot d$) | `src/features/biological_features.py` | ⚠️ **DERIVATIVE OF ADJACENT PROFILES** | Hadamard product of adjacent herb and selected drug vectors. No pair-level outcome override, but reflects profile curation. |
| `overlap_total` | `benchmarks/run_verified_hdi_benchmark.py` | ⚠️ **CURATION-SENSITIVE MECHANISTIC SIGNAL** | Captures whether an herb was curated as a modulator; survives herb-disjoint CV on this 18-herb panel (0.840 AUC), but cannot claim general botanical generalization. |
| `outcome` (Labels) | Benchmark CSV | ⚠️ **PER-STUDY FINDING, NOT CONSENSUS** | Reflects individual study conclusions rather than consensus pair ground-truth. Task A and Task B must be kept formally decoupled. |

---

## 8. Audit Sign-Off
Phase 0b audit revisions are complete. All claims of "zero leakage", "proves", and "mechanistically sound" have been removed and replaced with empirical, measured characterizations. Strict disjoint partition assertions are implemented in code and tested.
