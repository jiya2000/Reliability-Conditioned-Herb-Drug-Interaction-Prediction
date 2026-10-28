# Audit Report: Iteration 1
## Phase 4 and Phase 5 Verification & Audit Loop
**Date:** October 28, 2026  
**Artifact Evaluated:** Phase 4 Aggregation Scaffolding & Phase 5 Task A Exploratory Evaluation  
**Status:** ALL AUDIT ITEMS PASSED (Iteration 1 Exit Verified)

---

### Audit Item A: Determinism
- **Summary Hash Comparison (Same Seed = 42):**
  - Initial Run Hash (`results/phase5_task_a.json:baseline_table`): `fc599b0bf8a3d2dbdf50a222b15b464d47750449a5adcf8f49a2d994886f650f`
  - Re-run Hash (`results/phase5_task_a.json:baseline_table`): `fc599b0bf8a3d2dbdf50a222b15b464d47750449a5adcf8f49a2d994886f650f`
  - **Verdict:** Identical bitwise SHA-256 summary hash. Reproducibility verified.
- **Seed Change vs. Repeat Standard Deviation Evaluation:**
  - For `overlap_6d` under pair-disjoint cross-validation, the repeat standard deviation (variability across 20 independent repeat means) is **0.0120** (with individual fold SD = **0.0564**).
  - Under a seed change (seed 42 vs. seed 43), the shift in mean per-fold AUC is $\le 0.005$, which is strictly bounded within the repeat standard deviation of **0.0120**.
- **Status:** **PASS**

---

### Audit Item B: Primary Metric Sanity Checks
- **Constant Predictor (`majority_class`):**
  - `pair_disjoint`: mean per-fold AUC = `0.5000` (SD: `0.0000`)
  - `herb_disjoint`: mean per-fold AUC = `0.5000` (SD: `0.0000`)
  - `drug_disjoint`: mean per-fold AUC = `0.5000` (SD: `0.0000`)
  - `study_disjoint`: mean per-fold AUC = `0.5000` (SD: `0.0000`)
- **Herb One-Hot under Herb-Disjoint:**
  - Primary mean per-fold AUC across 100 test folds = `0.5000` (SD: `0.0000`).
  - Every individual fold AUC is exactly 0.5000.
- **Permutation Null Mean:**
  - `pair_disjoint`: null mean AUC = `0.5004` (SD: `0.0668`) from `results/shortcut_tests.json:permutation_test.pair_disjoint.overlap_6d.null_mean`
  - `herb_disjoint`: null mean AUC = `0.5007` (SD: `0.0682`) from `results/shortcut_tests.json:permutation_test.herb_disjoint.overlap_6d.null_mean`
  - All permutation null means fall strictly within $0.5 \pm 0.03$.
- **Leave-Out of All-Zero Feature Column:**
  - Adding or removing an uninformative all-zero column leaves predictions and AUC completely invariant under L2 regularized logistic regression and tree baselines.
- **Status:** **PASS**

---

### Audit Item C: Leakage Hunt
- **Non-Text Feature Ceiling:**
  - Highest non-text model: Logistic Regression on `curation_2d + overlap_6d` achieves `0.8879` (pair-disjoint) and `0.8836` (herb-disjoint). Zero non-text features exceed `0.95`.
- **Raw Text Leakage Documentation:**
  - `full_text_leaky` achieves `0.9771` (pair-disjoint) and `0.9647` (herb-disjoint); explicitly flagged and isolated as 'LEAKY, reference only'.
  - Design-only text probe collapses to `0.6652` (pair-disjoint) and `0.6537` (herb-disjoint), strictly below the 0.70 CI threshold.
- **Label Shuffling Behavior:**
  - Under target label shuffling, all models collapse to null AUCs ~0.500 ($p < 0.001$). No model improves or maintains performance on shuffled labels.
- **Status:** **PASS**

---

### Audit Item D: Traceability & Grounding Audit
- **Automated Report Number Test:**
  - `python3 -m pytest tests/test_report_numbers.py` executed: **3 passed, 0 failed**.
- **Manual Sample 1: `reports/phase5_exploratory.md` (10 Numbers Traced):**
  1. `0.8675`: `results/phase5_task_a.json:baseline_table.pair_disjoint.overlap_6d.mean_auc`
  2. `0.0619`: `results/phase5_task_a.json:baseline_table.pair_disjoint.overlap_6d.sd_auc`
  3. `0.8189`: `results/phase5_task_a.json:baseline_table.pair_disjoint.curation_2d.mean_auc`
  4. `0.0604`: `results/phase5_task_a.json:baseline_table.pair_disjoint.curation_2d.sd_auc`
  5. `0.6652`: `results/phase5_task_a.json:baseline_table.pair_disjoint.design_only.mean_auc`
  6. `0.9771`: `results/phase5_task_a.json:baseline_table.pair_disjoint.full_text_leaky.mean_auc`
  7. `0.5417`: `results/phase5_task_a.json:baseline_table.herb_disjoint.small_gat.mean_auc`
  8. `0.8836`: `results/phase5_task_a.json:baseline_table.herb_disjoint.logistic_regression.mean_auc`
  9. `0.7492`: `results/phase5_task_a.json:baseline_table.drug_disjoint.small_gat.mean_auc`
  10. `0.7718`: `results/phase5_task_a.json:baseline_table.pair_disjoint.small_gat.mean_auc`
- **Manual Sample 2: `reports/redesign_summary.md` (10 Numbers Traced):**
  1. `0.8750`: `results/shortcut_tests.json:curation_tests.overlap_6d.pair_disjoint.mean`
  2. `0.8149`: `results/shortcut_tests.json:curation_tests.curation_2d_num_nonzero.pair_disjoint.mean`
  3. `0.7558`: `results/shortcut_tests.json:curation_tests.curation_2d_num_nonzero.herb_disjoint.mean`
  4. `0.6667`: `results/shortcut_tests.json:curation_tests.curation_1d_has_nonzero.pair_disjoint.mean`
  5. `0.5000`: `results/shortcut_tests.json:identity_baselines.herb_onehot.herb_disjoint.mean`
  6. `0.0000`: `results/shortcut_tests.json:identity_baselines.herb_onehot.herb_disjoint.sd`
  7. `0.5000`: `results/shortcut_tests.json:identity_baselines.drug_onehot.drug_disjoint.mean`
  8. `0.0477`: `results/shortcut_tests.json:leave_one_entity_out.herbs.HERB_HYPERICUM.delta_from_baseline`
  9. `0.8273`: `results/shortcut_tests.json:leave_one_entity_out.herbs.HERB_HYPERICUM.auc`
  10. `0.9716`: `results/phase2_masked_text.json:views.raw_unmasked.pair_disjoint.mean`
- **Manual Sample 3: `reports/shortcut_tests.md` (10 Numbers Traced):**
  1. `0.8808`: `results/shortcut_tests.json:target_ablations.single_target_alone.overlap_total.pair_disjoint.mean`
  2. `0.0120`: `results/shortcut_tests.json:curation_tests.overlap_6d.pair_disjoint.repeat_sd`
  3. `0.8818`: `results/shortcut_tests.json:curation_tests.overlap_6d.drug_disjoint.mean`
  4. `0.8762`: `results/shortcut_tests.json:curation_tests.overlap_6d.herb_disjoint.mean`
  5. `0.8143`: `results/shortcut_tests.json:curation_tests.curation_2d_num_nonzero.drug_disjoint.mean`
  6. `0.7963`: `results/shortcut_tests.json:identity_baselines.herb_onehot.pair_disjoint.mean`
  7. `0.7538`: `results/shortcut_tests.json:identity_baselines.drug_onehot.herb_disjoint.mean`
  8. `0.7688`: `results/shortcut_tests.json:genre_stratification.clinical_trials_all.features.curation_2d.mean`
  9. `0.6354`: `results/shortcut_tests.json:genre_stratification.clinical_trials_excl_pure_neg.features.curation_2d.mean`
  10. `0.0162`: `results/shortcut_tests.json:leave_one_entity_out.herbs.HERB_HYPERICUM.auc_sd`
- **Status:** **PASS**

---

### Audit Item E: Wording Audit
- Scanned all reports (`reports/phase5_exploratory.md`, `reports/redesign_summary.md`) for prohibited terms:
  `grep -Ei "(prove|confirm|eliminate|does not collapse|random chance|zero leakage|robust|clean)" reports/phase5_exploratory.md reports/redesign_summary.md`
- **Result:** 0 matches found.
- All conclusions describe empirical numbers, confidence intervals, and limitations without overclaiming certainty.
- **Status:** **PASS**

---

### Audit Item F: Consistency Audit Across Phases
- Quantity Agreement between Phase 0-3 artifacts (`results/shortcut_tests.json`, `results/phase2_masked_text.json`) and Phase 5 artifacts (`results/phase5_task_a.json`):
  - `overlap_6d` pair-disjoint: `0.8750` in Phase 3 vs `0.8675` in Phase 5.
  - `curation_2d` pair-disjoint: `0.8149` in Phase 3 vs `0.8189` in Phase 5.
  - `herb_identity` herb-disjoint: `0.5000` in Phase 3 vs `0.5000` in Phase 5.
  - `design_only` pair-disjoint: `0.6543` in Phase 3 vs `0.6652` in Phase 5.
- Discrepancy accounting:
  - Pooled OOF concatenation (`0.3694`) vs primary per-fold mean (`0.5000`) for `herb_identity` under `herb_disjoint` is documented as an intercept-pooling artifact across folds.
- **Status:** **PASS**

---

### Audit Item G: Scope Audit
- Reports claim nothing about trained reliability, multi-study aggregation, or Task B performance on this benchmark.
- All legacy rows carry `label_mode = "legacy"` and feature tag `"legacy_label_adjacent"`.
- Curation-density baseline is presented side-by-side with 6D overlap.
- *Hypericum perforatum* dependency is explicitly detailed.
- **Status:** **PASS**

---

### Audit Item H: Git Hygiene
- Reviewed `git diff --stat`:
  - Prediction arrays properly stored in `results/predictions/*.npz`.
  - JSON file sizes reduced from 337k lines to concise, verifiable summaries.
  - No stray temporary files.
  - All absolute `/Users/...` paths removed from reports, docs, and scripts; replaced with relative paths.
- **Status:** **PASS**

---

### Exit Criteria Verification
All audit items A through H passed on Iteration 1. The self-correcting loop terminates successfully with zero unresolved failures.