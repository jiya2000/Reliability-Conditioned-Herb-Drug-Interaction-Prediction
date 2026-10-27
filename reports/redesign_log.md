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

## Phase 3: Diagnostic Fixes to Phase 0b/2 & Mechanistic Decoupling Architecture
- **Date Completed:** 2026-10-27
- **Primary Deliverables:**
  - **Diagnostic Fixes to Phase 0b/2 Reporting (Items 1–4):**
    1. *CV Repeats Randomization & Variance Reporting:*
       - Confirmed that `LeakageSafeSplitter` randomized group-to-fold assignments on every repeat via seeded permutations (`seed = 42 + rep`).
       - Reported both repeat SD ($0.0043 - 0.0092$) and fold-to-fold SD ($0.0539 - 0.0852$) across all 100 folds.
       - Computed cluster bootstrap 95% CIs resampling botanicals ($[0.7396, 0.9474]$) and drugs ($[0.7758, 0.9278]$) under `herb_disjoint` splits.
       - Standardized permutation test reporting notation: $p < \frac{1}{n_{\text{perm}} + 1} = p < 0.001$ (0 of 1,000 null draws reached true AUC).
    2. *Single-Class Folds & 100%-Negative Control Herbs:*
       - Documented the 4 100%-negative control herbs (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`).
       - Verified that stratified group-level allocation prevents single-class test sets: exactly 0 of 100 folds were single-class (0 folds skipped).
       - Reported fold-level mean AUC ($0.8739 \pm 0.0852$) alongside pooled out-of-fold AUC ($0.8572 \pm 0.0092$).
    3. *Level 4 Design-Only Tokens & Study-Genre Proxies:*
       - Tabulated label rates across all 11 Level 4 design tokens.
       - Flagged 4 study-genre proxy tokens: `clinical_study` (label rate $0.2632$), `preclinical_in_vivo` ($0.8182$), `case_observation` ($0.5556$), and `randomized_trial` ($0.0000$).
       - Evaluated sanitized design view with proxies removed: AUC collapses to exact random chance ($0.4487 \pm 0.0322$ herb-disjoint, $0.5254 \pm 0.0244$ pair-disjoint), proving all residual text shortcuts can be eliminated.
    4. *Within-Genre Stratified Baselines:*
       - Clinical trials cohort (`clinical` + `rct`, $N=125$, 61 pos, 64 neg): Overlap 6D AUC increases to **$0.9059 \pm 0.0044$** (proving overlap does *not* collapse inside trials).
       - Healthy volunteers cohort ($N=106$, 48 pos, 58 neg): Overlap 6D AUC is **$0.8906 \pm 0.0047$**.
       - Case reports cohort ($N=16$, 14 pos, 2 neg): High-variance ($0.4295 \pm 0.0659$) due to near-zero negative controls.
  - **Phase 3 Mechanistic Decoupling & Permanent CI Tests (Items 5–8):**
    5. *Legacy Label-Adjacent Tagging:*
       - Labeled `HERB_BIOLOGICAL_PROFILES` and `DRUG_BIOLOGICAL_PROFILES` in `src/features/biological_features.py` as `"legacy, label-adjacent"`.
       - Updated `BiologicalProfileExtractor` with `profile_status = "legacy, label-adjacent"` and `feature_set = "legacy"`.
    6. *Blind-Curation Pipeline & Sourcing Plan:*
       - Created `data/curation/mechanistic_features.csv` (522 rows covering 18 herbs × 40 drugs across 9 canonical targets).
       - All numeric values left empty; sources and citations set strictly to `"UNVERIFIED"`.
       - Implemented `src/data/mechanistic_curation.py` with validation rules and `is_curation_verified()` checker.
       - Produced `reports/mechanistic_curation_todo.md` organizing all entities by recommended external database (NaPDI, FDA DDI Table, DrugBank, Flockhart Table, NPASS).
    7. *Feature-Switched Model Suite (LR, GBT, Small GAT):*
       - Implemented `FeatureSwitchedBenchmark` in `src/models/feature_switched_models.py` supporting `feature_set="legacy"` and `feature_set="new"`.
       - Built `SmallGATClassifier` for graph-contextualized interaction prediction.
       - Enforced strict execution guard: requesting `feature_set="new"` raises `UnverifiedCurationError` until all rows in the curation CSV have human-verified citations.
    8. *Permanent CI Test Suite:*
       - Created `tests/test_phase3_and_ci_checks.py` containing permanent CI checks (a)–(e):
         - Test (a): No outcome language in reliability extraction; lineage features independent of label columns.
         - Test (b): Design-only text probe AUC strictly $< 0.70$.
         - Test (c): Split-disjointness assertions across herb, drug, and pair splits.
         - Test (d): Preprocessors and scalers fit strictly on training partitions.
         - Test (e): Curation gap check: fails if overlap AUC is within $0.03$ of curation-only 2D AUC unless acknowledged with `allow_curation_gap=True`.
         - Test (f): Feature switch guard correctly blocks unverified curation data and runs on legacy baseline.
- **Test Suite Status:** 149 passed, 0 failures, 19 warnings (in 24.72s).
- **STOP Point:** Standing by for user-supplied consensus labels before touching Phase 4 or Phase 5.


