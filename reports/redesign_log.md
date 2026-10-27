# ARCHITECTURAL REDESIGN & EVIDENCE DECOUPLING LOG
## Reliability-Conditioned Herb–Drug Interaction (HDI) Prediction Framework

This log tracks every architectural change, feature provenance audit, schema decoupling, and quantitative benchmark shift across the core redesign.

---

## Phase 0: Provenance Audit & Disjoint Cross-Validation Probes
- **Date Completed:** 2026-10-27
- **Primary Deliverables:**
  - `reports/provenance_audit.md`: Exhaustive audit of `BiologicalProfileExtractor`, `CANONICAL_TARGETS`, `HERB_BIOLOGICAL_PROFILES`, `DRUG_BIOLOGICAL_PROFILES`, `interaction_vec`, and `overlap_total`.
  - Disjoint Cross-Validation Probe Suite (`GroupKFold` 5-fold CV):
    - Evaluated `overlap_6d`, `quality_7d`, `combined_13d`, and `herb_onehot` across 3 strict partitionings: `pair-disjoint`, `drug-disjoint`, and `herb-disjoint`.
- **Quantitative Results Recorded:**

| Feature Representation | `pair-disjoint` CV AUC | `drug-disjoint` CV AUC | `herb-disjoint` CV AUC |
| :--- | :---: | :---: | :---: |
| **6D Biological Overlap** | **0.868 ± 0.022** | **0.863 ± 0.090** | **0.840 ± 0.098** |
| **Combined 13D** | **0.883 ± 0.032** | **0.880 ± 0.081** | **0.862 ± 0.092** |
| **7D Quality Lineage** | 0.652 ± 0.076 | 0.642 ± 0.117 | 0.658 ± 0.019 |
| **Herb One-Hot Only** | 0.800 ± 0.125 | 0.792 ± 0.058 | **0.500 ± 0.000** |

- **Key Scientific Findings:**
  1. **Pharmacokinetic Generalization Confirmed:** 6D biological overlap features maintain an AUC of **$0.840 \pm 0.098$ even on completely unseen botanicals** (`herb-disjoint`), proving that pharmacokinetic overlap is not merely a botanical entity shortcut.
  2. **Mathematical Independence of `interaction_vec`:** Verified that `interaction_vec` is strictly the Hadamard product $h_i \cdot d_i$ between independent single-entity vectors, with no pair-level empirical lookup table or outcome column leakage.
  3. **Zero Split Contamination Confirmed:** Herb one-hot identity collapses to exactly $0.500 \pm 0.000$ under `herb-disjoint` splitting, verifying strict partition integrity.

---

## Phase 1: Separation of Evidence from Ground Truth
- **Date Completed:** 2026-10-27
- **Primary Deliverables:**
  - `src/data/evidence_schema.py`:
    - Added `ConsensusLabel` Enum (`interaction`, `no_interaction`, `disputed`).
    - Added `ConsensusPairRecord` dataclass and `validate_consensus_labels_table()`.
    - Added `load_consensus_labels_table()` with prominent `[LEGACY LABEL MODE ACTIVE]` logging when unpopulated.
    - Added `StudyFinding` Enum and `extract_study_design_fields()` to isolate study methodology (species, assay, route, preparation, dose, sample size $n$) from study conclusions.
  - `data/benchmark/consensus_labels.csv`: Created canonical pair-level ground-truth template awaiting external compendium curation.
  - `docs/task_definition.md`: Formally formulated:
    - **Task A (Mechanistic Cold-Start):** Pure pharmacokinetic prediction on unstudied pairs with zero literature and zero text.
    - **Task B (Supervised Evidence Aggregation):** Residual architecture $\text{logit}_{\text{fused}} = \text{logit}_{\text{prior}} + \sum R_i \cdot \delta_i$ where reliability $R_i$ is supervised by study agreement with external consensus.
- **Test Suite Status:** All unit and regression tests passing.

---

## Phase 0b: Audit Revision, Code Hygiene & Shortcut Tests
- **Date Completed:** 2026-10-27
- **Primary Deliverables:**
  - `reports/provenance_audit.md` (REV2):
    - Reclassified `HERB_BIOLOGICAL_PROFILES` as **LABEL-ADJACENT (pending per-value citation)**.
    - Reclassified `DRUG_BIOLOGICAL_PROFILES` as **pathway values independent of labels, but drug selection is label-adjacent**.
    - Retracted claim that `herb_onehot = 0.500` under herb-disjoint proves zero leakage; replaced with code assertions (`assert len(set(train_df[entity]) & set(test_df[entity])) == 0`) and unit tests in `tests/test_audit_and_leakage_checks.py`.
    - Added comprehensive Limitations section: (a) negative control herbs (Valerian, Saw Palmetto, Milk Thistle) are label-level concepts in profile dictionaries; (b) curation density shortcut explains substantial signal; (c) audit written by internal team; (d) fold SD 0.08–0.12.
    - Documented 141-record panel selection: deliberate negative controls (4 exclusive 100% negative control herbs: Black Cohosh, Milk Thistle, Saw Palmetto, Valerian), with complete 18-herb and 40-drug breakdown.
  - `reports/profile_provenance.csv`:
    - Full per-value table (111 rows: 51 herb targets, 60 drug targets).
    - Source citations: 100% `UNVERIFIED` (code comments contain qualitative phrases, no formal literature DOIs/PMIDs).
    - Written before seeing labels: 100% `no` (git commits prove `data/benchmark/verified_botanical_hdi.csv` existed in `8ebd18c` at 11:32:49, prior to `src/features/biological_features.py` in `79b1ada` at 11:57:12).
  - Code Hygiene in `benchmarks/run_verified_hdi_benchmark.py`:
    - Replaced `TARGET_TO_IDX.get(name, <int>)` with strict `TARGET_TO_IDX[name]` across lines 70–74.
    - Added unit test `test_target_index_hygiene_raises_key_error` asserting missing keys raise `KeyError`.
    - Verified extraction unchanged: before/after shape `(141, 18)`, identical overlap values.
  - Shortcut and Robustness Tests (20 Repeats of 5-Fold Grouped CV, Logistic Regression):
    - Documented in `reports/shortcut_tests.md` and `results/shortcut_tests.json`.
    - **Curation-Only Features:**
      - `curation_1d_has_nonzero`: herb-disjoint `0.5385 ± 0.0334`, pair-disjoint `0.6057 ± 0.0132`.
      - `curation_2d_num_nonzero`: herb-disjoint `0.7984 ± 0.0151`, pair-disjoint `0.8016 ± 0.0061` (within ~0.06 of `overlap_6d` AUC, establishing that ~0.80 AUC is driven by profile curation density).
    - **Per-Target Ablations:**
      - `overlap_total` is the dominant feature: alone it achieves `0.8695` pair-disjoint AUC; removing it causes performance to drop to `0.7720` (pair-disjoint) and `0.6916` (drug-disjoint).
      - `CYP3A4` is the second strongest single target (`0.6892` pair-disjoint alone).
    - **Permutation Test (1000 Shuffles):**
      - Null AUC distribution: mean `0.4526 ± 0.0657` (pair-disjoint) / `0.4555 ± 0.0649` (herb-disjoint).
      - Empirical p-value = `0.000000` for both `overlap_6d` and `combined_13d`.
    - **Per-Entity Leave-One-Out:**
      - No single entity drops AUC by $> 0.05$ (Hypericum drops AUC by `0.0470`, from `0.8669` to `0.8199`).
    - **Entity-Identity Baselines:**
      - `herb_onehot`: `0.7874 ± 0.0177` (pair-disjoint).
      - `drug_onehot`: `0.7166 ± 0.0166` (pair-disjoint).

## Phase 2: Remove Text and Identity Shortcuts
- **Date Completed:** 2026-10-27
- **Primary Deliverables:**
  - `src/features/text_masking.py`:
    - Implemented `mask_outcome_language`: masks directional alteration verbs, statistical conclusions, outcome percentages, and clinical toxicity phrases.
    - Implemented `mask_entities`: masks all 18 botanical names, Latin binomials, active constituent tokens, and 40 prescription drug names.
    - Implemented `mask_outcome_and_entities` (with optional clinical proxy masking).
    - Implemented `extract_study_design_text`: extracts strictly methodological and study-design tokens (`randomized_trial`, `clinical_study`, `oral_administration`, `standardized_extract`), discarding free-form sentences.
  - `tests/test_text_masking.py`: Unit test suite verifying masking of positive phrases, negative phrases, entity names, combined masking, proxy masking, design-only extraction, and edge cases. (7/7 passed).
  - Systematic Text Probe Evaluation across 4 Masking Levels (20 Repeats of 5-Fold Grouped CV):
    - Documented in `reports/phase2_masked_text_report.md` and `results/phase2_masked_text.json`.

| Text Representation Level | `herb_disjoint` CV AUC | `drug_disjoint` CV AUC | `pair_disjoint` CV AUC | Status vs 0.70 Target |
| :--- | :---: | :---: | :---: | :---: |
| **0. Raw Unmasked Text** | 0.9476 ± 0.0056 | 0.9727 ± 0.0047 | 0.9716 ± 0.0065 | 🚨 Leaky Baseline |
| **1. Outcome Language Masked** | 0.8577 ± 0.0096 | 0.8825 ± 0.0106 | 0.9056 ± 0.0101 | ⚠️ Above 0.70 |
| **2. Outcome + Entities Masked** | 0.8305 ± 0.0108 | 0.7986 ± 0.0130 | 0.8464 ± 0.0129 | ⚠️ Above 0.70 |
| **3. Aggressive (with Context Proxies)** | 0.7950 ± 0.0130 | 0.7563 ± 0.0202 | 0.8208 ± 0.0143 | ⚠️ Above 0.70 |
| **4. Structured Design-Only View** | **0.6364 ± 0.0267** | **0.6237 ± 0.0277** | **0.6543 ± 0.0166** | ✅ **Safely Below 0.70** |

- **Root Cause Analysis (Why Levels 1–3 Remained Above 0.70):**
  - In this 141-record benchmark, 100% of negative control studies were pharmacokinetic trials in healthy volunteers (`"in healthy subjects"`: negative coefficient $-1.0216$).
  - In contrast, positive interaction findings frequently originated from case reports in patient cohorts (`"heart transplant recipients"`, `"patients"`: positive coefficient $+0.6180$).
  - As long as free-form abstract sentences remain, models exploit study genre/population as a label proxy. Discarding free-form abstract sentences and retaining strictly structured design tokens (Level 4) successfully brings text signal down to **0.6237 – 0.6543 AUC**.

---

## STOP Point (Awaiting User Guidance before Phase 3):
- Phase 0b (Audit revision, code hygiene, 20-repeat shortcut tests) and Phase 2 (Text masking, design-only view, probe re-evaluation) are complete.
- Standing by for user review and decision regarding the label-adjacent herb profiles before starting Phase 3.

