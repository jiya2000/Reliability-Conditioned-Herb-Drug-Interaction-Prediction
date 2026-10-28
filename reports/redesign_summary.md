# COMPREHENSIVE ARCHITECTURAL REDESIGN SUMMARY (PHASE 6)
## Complete Audit of Surviving, Softened, and Removed Claims

> [!IMPORTANT]
> **EVIDENCE AUDIT VERDICT:** The original pipeline's headline metrics were inflated by three intersecting shortcuts: (1) severe natural language outcome leakage in clinical abstract text ($0.97+$ AUC), (2) manual post-hoc profile curation density confounds where nonzero targets served as outcome proxies ($0.80+$ AUC), and (3) entity composition and baseline base-rate imbalances. Below is the audited accounting of all scientific claims rendered strictly from JSON test artifacts.

---

## 1. Claims Removed / Retracted
1. **Retracted: Single-Run Zero-Contamination Claim via 0.500 Herb One-Hot.**
   - **Finding:** In the initial single-run evaluation, herb one-hot under herb-disjoint CV was reported as exactly `0.500`. Under repeated grouped CV (20 repeats), the primary mean per-fold AUC is **0.5000** (SD: `0.0000`), while pooled out-of-fold concatenation across folds with differing intercepts dropped to `0.3670`. The earlier claim is formally withdrawn.
2. **Retracted: Pharmacokinetic Generalization Claim Without Curation Confound Qualification.**
   - **Finding:** The original claim asserted that 6D overlap demonstrated generalized mechanistic signal. However, a 2D curation density proxy (`has_measured_target`, `n_nonzero_targets`) alone achieves **0.7558** herb-disjoint AUC and **0.8149** pair-disjoint AUC without utilizing target identities. The claim of pure pharmacokinetic generalization is withdrawn.
3. **Retracted: SmallGAT Superiority.**
   - **Finding:** SmallGAT failed to demonstrate statistically significant superiority over Logistic Regression across all three disjoint protocols ($p \ge 0.05$) and is formally dropped.
4. **Retracted: Empirical Permutation p-value of 0.000000.**
   - **Finding:** Retracted literal float string notation. Formally bound to $p < 1 / (N_{\text{perm}} + 1) = p < 0.001$.

## 2. Claims Softened / Re-Contextualized
1. **Tabular Biological Overlap Performance:**
   - **Claim Softened:** 6D overlap achieves mean per-fold AUC of **0.8750** (pair-disjoint), **0.8818** (drug-disjoint), and **0.8762** (herb-disjoint). However, this signal is label-adjacent: botanical profiles were hand-curated in one file by the same team after benchmark labels were known, rendering all profile features unverified proxies.
2. **Study-Genre & Pure-Negative Sensitivity:**
   - **Claim Softened:** In the clinical trials cohort ($N=125$), `overlap_6d` scores **0.9027**, but when the 4 pure-negative botanicals are excluded ($N=100$), `curation_2d` drops from **0.7688** to **0.6354**, revealing that negative controls heavily anchor the profile density shortcut.
3. **Hypericum perforatum Dependency:**
   - **Claim Softened:** Leaving out *Hypericum perforatum* causes a performance drop of **0.0477** AUC (to **0.8273**), demonstrating that benchmark performance is disproportionately anchored by St. John's Wort.
4. **Residual Study-Genre Leakage in Design-Only Representation & Unexplained 0.4328 Result:**
   - **Claim Softened:** The Level 4 design-only representation extracts study type, route, and formulation tokens (specifically the `clinical_study` token triggered by 'volunteers' or 'subjects', and the `case_observation` token triggered by 'case report' or 'patient'; note that no explicit sample size tokens are extracted). In the clinical trials cohort ($N=125$), the sanitized design-only view (with genre proxies removed) drops to **0.4743**, and when pure-negative controls are excluded ($N=100$), drops further to **0.4328**. This 0.4328 sub-chance result in the clinical trials cohort ($N=100$, 61 positive, 39 negative, positive base rate = 0.6100) is **unexplained**. Empirical token label rates in this cohort are: herbal_tea (N=6, 66.7% pos), intravenous_administration (N=28, 60.7% pos), juice_preparation (N=9, 44.4% pos), methodology_unspecified (N=45, 66.7% pos), oil_formulation (N=4, 75.0% pos), oral_administration (N=10, 70.0% pos), standardized_extract (N=14, 35.7% pos).

## 3. Claims That Survive (Rigorous & Traceable)
1. **Severe Text Outcome Leakage:**
   - Raw unmasked clinical study text achieves an AUC of **0.9716** (pair-disjoint) and **0.9727** (drug-disjoint), showing that clinical abstracts reflect study findings in natural language representations.
2. **Task B Non-Testability on Current Benchmark:**
   - On the 141-record panel, exactly 15 of 120 unique pairs have $\ge 2$ independent studies (~1 study per pair). Task B multi-study evidence aggregation cannot be tested until multi-study consensus benchmarks are curated.

---
### 4. Benchmark Architectural & Artifact Summary
| Metric / Artifact | Audited Value | Source JSON Path |
| :--- | :---: | :--- |
| Raw Text Pair-Disjoint AUC | `0.9716` | `results/phase2_masked_text.json:views.raw_unmasked.pair_disjoint.mean` |
| Design-Only Pair-Disjoint AUC | `0.6543` | `results/phase2_masked_text.json:views.level4_design_only.pair_disjoint.mean` |
| Overlap 6D Pair-Disjoint AUC | `0.8750` | `results/shortcut_tests.json:curation_tests.overlap_6d.pair_disjoint.mean` |
| Curation 2D Pair-Disjoint AUC | `0.8149` | `results/shortcut_tests.json:curation_tests.curation_2d_num_nonzero.pair_disjoint.mean` |
| Curation 1D Pair-Disjoint AUC | `0.6667` | `results/shortcut_tests.json:curation_tests.curation_1d_has_nonzero.pair_disjoint.mean` |
| Herb One-Hot Herb-Disjoint AUC | `0.5000` | `results/shortcut_tests.json:identity_baselines.herb_onehot.herb_disjoint.mean` |
| Drug One-Hot Drug-Disjoint AUC | `0.5000` | `results/shortcut_tests.json:identity_baselines.drug_onehot.drug_disjoint.mean` |
| Hypericum Drop from Baseline | `0.0477` | `results/shortcut_tests.json:leave_one_entity_out.herbs.HERB_HYPERICUM.delta_from_baseline` |
