# Shortcut and Robustness Test Report
## Reliability-Conditioned Herb–Drug Interaction (HDI) Benchmark

**Document Identifier:** `REP-SHORTCUT-TESTS-2026-10-28`  
**Protocol:** 20 repeats of 5-fold grouped CV, Logistic Regression, cluster bootstrap CIs.  
**Primary Metric:** Mean per-fold ROC-AUC (5 folds x 20 repeats = 100 fold evaluations).  
**Secondary Metric:** Pooled out-of-fold ROC-AUC across folds.  
**Raw Results:** [results/shortcut_tests.json](results/shortcut_tests.json)

---

## 1. Curation-Only Features vs Overlap-6D

Comparison of mechanistic overlap (`overlap_6d`) against simple curation profile density features:
- **`overlap_6d`**: 6D pharmacokinetic interaction vector `[cyp3a4, pgp, cyp2c9, cyp1a2, cyp2e1, overlap_total]`.
- **`curation_1d_has_nonzero`**: Binary indicator whether the botanical entity has any target with modulatory value $> 0.05$.
- **`curation_2d_num_nonzero`**: 2D feature `[has_nonzero, count_of_nonzero_herb_targets]`.

| Feature Representation | Split Mode | Mean Per-Fold AUC | Repeat SD | Fold SD | Herb Cluster Bootstrap 95% CI | Drug Cluster Bootstrap 95% CI | Pooled OOF AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`overlap_6d`** | `herb_disjoint` | 0.8762 | 0.0153 | 0.0878 | [0.7983, 0.9283] | [0.8084, 0.9321] | 0.8571 |
| `curation_1d_has_nonzero` | `herb_disjoint` | 0.6372 | 0.0244 | 0.1473 | [0.5000, 0.6955] | [0.6101, 0.6740] | 0.5414 |
| `curation_2d_num_nonzero` | `herb_disjoint` | 0.7558 | 0.0388 | 0.1324 | [0.6151, 0.7908] | [0.7055, 0.8173] | 0.7985 |
| **`overlap_6d`** | `drug_disjoint` | 0.8818 | 0.0139 | 0.0753 | [0.7672, 0.9531] | [0.8080, 0.9303] | 0.8559 |
| `curation_1d_has_nonzero` | `drug_disjoint` | 0.6548 | 0.0333 | 0.0939 | [0.5000, 0.8019] | [0.5884, 0.7112] | 0.5643 |
| `curation_2d_num_nonzero` | `drug_disjoint` | 0.8143 | 0.0277 | 0.0862 | [0.6267, 0.9173] | [0.7212, 0.8608] | 0.7988 |
| **`overlap_6d`** | `pair_disjoint` | 0.8750 | 0.0120 | 0.0564 | [0.7654, 0.9481] | [0.8076, 0.9367] | 0.8671 |
| `curation_1d_has_nonzero` | `pair_disjoint` | 0.6667 | 0.0240 | 0.1089 | [0.5000, 0.8128] | [0.6085, 0.7231] | 0.6045 |
| `curation_2d_num_nonzero` | `pair_disjoint` | 0.8149 | 0.0098 | 0.0784 | [0.6625, 0.9172] | [0.7565, 0.8714] | 0.8009 |

### Key Findings on Curation and Overlap:
- Under `herb_disjoint` splits, `overlap_6d` achieves a mean per-fold AUC of **0.8762** with fold-to-fold SD of **0.0878** and pooled out-of-fold AUC of **0.8571**.
- `curation_2d_num_nonzero` achieves **0.7558** per-fold AUC on `herb_disjoint` and **0.8149** on `pair_disjoint`, demonstrating that botanical target profile curation density accounts for a substantial proportion of overall predictive capacity.

---

## 2. Per-Target Ablation and Nonzero Feature Distribution

### 2.1 Nonzero Value Distribution Across Overlap Columns
Analysis of feature density across the 6 overlap columns in the 141 benchmark records:

| Overlap Feature Column | Nonzero Records | Total Records | Fraction Nonzero | Min Value | Max Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CYP3A4` | 55 | 141 | 0.3901 (39.01%) | 0.0000 | 1.0000 |
| `ABCB1_Pgp` | 23 | 141 | 0.1631 (16.31%) | 0.0000 | 1.0000 |
| `CYP2C9` | 14 | 141 | 0.0993 (9.93%) | 0.0000 | 0.8000 |
| `CYP1A2` | 3 | 141 | 0.0213 (2.13%) | 0.0000 | 0.8000 |
| `CYP2E1` | 0 | 141 | 0.0000 (0.00%) | 0.0000 | 0.0000 |
| `overlap_total` | 92 | 141 | 0.6525 (65.25%) | 0.0000 | 2.0000 |

### Explanation of Near-Constant Columns:
- **`CYP2E1`** is entirely zero across all 141 records (0.0000 fraction nonzero). None of the benchmark botanicals or co-administered drugs possess active CYP2E1 modulatory overlap.
- **`CYP1A2`** is nonzero in only 3 out of 141 records (0.0213 fraction nonzero).
- **`CYP2C9`** is nonzero in only 14 out of 141 records (0.0993 fraction nonzero).
- The overwhelming majority of feature variance is concentrated in **`CYP3A4`** (39.01% nonzero), **`ABCB1_Pgp`** (16.31% nonzero), and the composite **`overlap_total`** (65.25% nonzero).

### 2.2 Single Target Alone

| Target Feature | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |
| :--- | :---: | :---: | :---: |
| `CYP3A4` | 0.6784 ± 0.0392 | 0.7204 ± 0.0354 | 0.7209 ± 0.0197 |
| `ABCB1_Pgp` | 0.5865 ± 0.0393 | 0.6089 ± 0.0370 | 0.6242 ± 0.0249 |
| `CYP2C9` | 0.5395 ± 0.0286 | 0.4657 ± 0.0297 | 0.5277 ± 0.0162 |
| `CYP1A2` | 0.5176 ± 0.0353 | 0.5183 ± 0.0296 | 0.5313 ± 0.0276 |
| `CYP2E1` | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |
| `overlap_total` | 0.8795 ± 0.0139 | 0.8842 ± 0.0142 | 0.8808 ± 0.0114 |

### 2.3 Leave-One-Target-Out

| Feature Removed | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |
| :--- | :---: | :---: | :---: |
| Without `CYP3A4` | 0.8797 ± 0.0139 | 0.8851 ± 0.0136 | 0.8787 ± 0.0099 |
| Without `ABCB1_Pgp` | 0.8744 ± 0.0145 | 0.8822 ± 0.0138 | 0.8760 ± 0.0108 |
| Without `CYP2C9` | 0.8777 ± 0.0147 | 0.8817 ± 0.0136 | 0.8764 ± 0.0129 |
| Without `CYP1A2` | 0.8764 ± 0.0156 | 0.8811 ± 0.0141 | 0.8749 ± 0.0112 |
| Without `CYP2E1` | 0.8762 ± 0.0153 | 0.8818 ± 0.0139 | 0.8750 ± 0.0120 |
| Without `overlap_total` | 0.7705 ± 0.0278 | 0.7336 ± 0.0264 | 0.7951 ± 0.0158 |

### Target Redundancy Mechanics:
- Removing `overlap_total` drops pair-disjoint AUC from 0.8750 to 0.7951 (and drops drug-disjoint from 0.8818 to 0.7336).
- Conversely, removing individual CYP enzymes like `CYP3A4` results in minor changes (0.8787 on pair-disjoint) because `overlap_total` already incorporates the CYP3A4 modulatory interaction sum.

---

## 3. Label Permutation Tests (1,000 Shuffles)

| Model / Feature | Split Mode | True Mean AUC | Null Mean ± SD | Null 95% Interval | Empirical p-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`overlap_6d`** | `herb_disjoint` | 0.8762 | 0.5007 ± 0.0682 | [0.3615, 0.6264] | **p < 1/(n_perm+1)** |
| **`combined_13d`** | `herb_disjoint` | 0.8893 | 0.5014 ± 0.0693 | [0.3682, 0.6317] | **p < 1/(n_perm+1)** |
| **`overlap_6d`** | `pair_disjoint` | 0.8750 | 0.5004 ± 0.0668 | [0.3682, 0.6273] | **p < 1/(n_perm+1)** |
| **`combined_13d`** | `pair_disjoint` | 0.8914 | 0.5005 ± 0.0672 | [0.3663, 0.6228] | **p < 1/(n_perm+1)** |

---

## 4. Per-Entity Leave-One-Out Sensitivity Analysis

Baseline Full `overlap_6d` Pair-Disjoint AUC: **0.8750**  
Sensitivity criterion: Flagged if entity removal drops pair-disjoint AUC by $> 0.05$.

### 4.1 Leave-One-Herb-Out

| Herb Removed | Records (N) | Remaining (N) | Mean AUC ± SD | Δ AUC | Flagged (>0.05 drop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `HERB_BLACKCOHOSH` | 5 | 136 | 0.8677 ± 0.0061 | +0.0073 | No |
| `HERB_CHAMOMILE` | 3 | 138 | 0.8794 ± 0.0135 | -0.0044 | No |
| `HERB_CRANBERRY` | 4 | 137 | 0.8748 ± 0.0138 | +0.0002 | No |
| `HERB_CURCUMA` | 8 | 133 | 0.8703 ± 0.0115 | +0.0047 | No |
| `HERB_ECHINACEA` | 8 | 133 | 0.8817 ± 0.0087 | -0.0067 | No |
| `HERB_GARLIC` | 9 | 132 | 0.9031 ± 0.0090 | -0.0281 | No |
| `HERB_GINKGO` | 15 | 126 | 0.8919 ± 0.0077 | -0.0169 | No |
| `HERB_GINSENG` | 10 | 131 | 0.8860 ± 0.0060 | -0.0110 | No |
| `HERB_GOLDENSEAL` | 7 | 134 | 0.8660 ± 0.0139 | +0.0090 | No |
| `HERB_GRAPEFRUIT` | 6 | 135 | 0.8804 ± 0.0064 | -0.0054 | No |
| `HERB_GREENTEA` | 10 | 131 | 0.8800 ± 0.0104 | -0.0050 | No |
| `HERB_HYPERICUM` | 21 | 120 | 0.8273 ± 0.0162 | +0.0477 | No |
| `HERB_KAVA` | 5 | 136 | 0.8881 ± 0.0091 | -0.0131 | No |
| `HERB_LICORICE` | 7 | 134 | 0.8789 ± 0.0099 | -0.0039 | No |
| `HERB_MILKTHISTLE` | 9 | 132 | 0.8696 ± 0.0077 | +0.0054 | No |
| `HERB_PEPPERMINT` | 3 | 138 | 0.8801 ± 0.0171 | -0.0051 | No |
| `HERB_SAWPALMETTO` | 5 | 136 | 0.8677 ± 0.0061 | +0.0073 | No |
| `HERB_VALERIAN` | 6 | 135 | 0.8773 ± 0.0090 | -0.0023 | No |

### 4.2 Leave-One-Drug-Out (Drugs with N >= 2)

| Drug Removed | Records (N) | Remaining (N) | Mean AUC ± SD | Δ AUC | Flagged (>0.05 drop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DRUG_WARFARIN` | 21 | 120 | 0.8944 ± 0.0076 | -0.0194 | No |
| `DRUG_DIGOXIN` | 15 | 126 | 0.8763 ± 0.0102 | -0.0013 | No |
| `DRUG_MIDAZOLAM` | 15 | 126 | 0.8834 ± 0.0149 | -0.0084 | No |
| `DRUG_DEXTROMETHORPHAN` | 12 | 129 | 0.8609 ± 0.0095 | +0.0141 | No |
| `DRUG_CYCLOSPORINE` | 7 | 134 | 0.8716 ± 0.0148 | +0.0034 | No |
| `DRUG_INDINAVIR` | 5 | 136 | 0.8698 ± 0.0080 | +0.0052 | No |
| `DRUG_THEOPHYLLINE` | 5 | 136 | 0.8714 ± 0.0128 | +0.0036 | No |
| `DRUG_TACROLIMUS` | 4 | 137 | 0.8761 ± 0.0061 | -0.0011 | No |
| `DRUG_ALPRAZOLAM` | 4 | 137 | 0.8748 ± 0.0082 | +0.0002 | No |
| `DRUG_ASPIRIN` | 4 | 137 | 0.8719 ± 0.0073 | +0.0031 | No |
| `DRUG_SIMVASTATIN` | 3 | 138 | 0.8792 ± 0.0091 | -0.0042 | No |
| `DRUG_ROSUVASTATIN` | 3 | 138 | 0.8793 ± 0.0085 | -0.0043 | No |
| `DRUG_NIFEDIPINE` | 3 | 138 | 0.8886 ± 0.0104 | -0.0136 | No |
| `DRUG_CHLORZOXAZONE` | 3 | 138 | 0.8968 ± 0.0080 | -0.0218 | No |
| `DRUG_SAQUINAVIR` | 2 | 139 | 0.8689 ± 0.0096 | +0.0061 | No |
| `DRUG_RITONAVIR` | 2 | 139 | 0.8798 ± 0.0087 | -0.0048 | No |
| `DRUG_NADOLOL` | 2 | 139 | 0.8771 ± 0.0090 | -0.0021 | No |
| `DRUG_PREDNISOLONE` | 2 | 139 | 0.8782 ± 0.0078 | -0.0032 | No |
| `DRUG_HYDROCHLOROTHIAZIDE` | 2 | 139 | 0.8784 ± 0.0133 | -0.0034 | No |
| `DRUG_DOCETAXEL` | 2 | 139 | 0.8712 ± 0.0101 | +0.0038 | No |
| `DRUG_OMEPRAZOLE` | 2 | 139 | 0.8752 ± 0.0089 | -0.0002 | No |
| `DRUG_IRINOTECAN` | 2 | 139 | 0.8772 ± 0.0088 | -0.0022 | No |
| `DRUG_METFORMIN` | 2 | 139 | 0.8819 ± 0.0101 | -0.0069 | No |
| `DRUG_CAFFEINE` | 2 | 139 | 0.8715 ± 0.0082 | +0.0035 | No |
| `DRUG_FELODIPINE` | 2 | 139 | 0.8711 ± 0.0104 | +0.0039 | No |

---

## 5. Entity-Identity-Only Baselines

Performance of categorical one-hot entity identity baselines under repeated grouped CV:

| Identity Representation | Split Mode | Mean Per-Fold AUC | Repeat SD | Fold SD | Pooled OOF AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`herb_onehot`** (18-D) | `herb_disjoint` | 0.5000 | 0.0000 | 0.0000 | 0.3694 |
| **`drug_onehot`** (40-D) | `herb_disjoint` | 0.7538 | 0.0240 | 0.1037 | 0.7049 |
| **`herb_onehot`** (18-D) | `drug_disjoint` | 0.8170 | 0.0255 | 0.0857 | 0.7948 |
| **`drug_onehot`** (40-D) | `drug_disjoint` | 0.5000 | 0.0000 | 0.0000 | 0.4065 |
| **`herb_onehot`** (18-D) | `pair_disjoint` | 0.7963 | 0.0176 | 0.0867 | 0.7874 |
| **`drug_onehot`** (40-D) | `pair_disjoint` | 0.7273 | 0.0232 | 0.0995 | 0.7167 |

### Mathematical Property of Per-Fold Primary Metric:
- Under `herb_disjoint` splits, all test botanical entities are unseen during training. A linear one-hot classifier cannot discriminate between test instances and predicts constant probabilities (`sigmoid(intercept)`). Consequently, the per-fold AUC is exactly **0.5000 ± 0.0000** across all folds.
- The lower pooled out-of-fold AUC (**0.3694**) arises because pooling out-of-fold predictions across folds with differing class prevalences creates rank-order distortions between folds.

---

## 6. Genre Stratification Under Herb-Disjoint Splits

Stratified evaluation within study design genres under `herb_disjoint` grouped cross-validation, evaluated both with and without the 4 pure-negative control botanicals (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`):

| Study Genre Cohort | Pure-Negatives Included | Cohort Size (N) | Class Balance | `overlap_6d` Mean AUC | `overlap_total` Mean AUC | `curation_2d` Mean AUC | `design_only` Mean AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clinical Trials Cohort (All) | Yes (All 18 Herbs) | 125 | 61 pos / 64 neg | 0.9027 ± 0.0266 | 0.9080 ± 0.0256 | 0.7688 ± 0.0312 | 0.4743 ± 0.0891 |
| Clinical Trials Cohort (Excl. 4 Pure-Negative Herbs) | No (14 Herbs) | 100 | 61 pos / 39 neg | 0.8779 ± 0.0189 | 0.8881 ± 0.0175 | 0.6354 ± 0.0374 | 0.4328 ± 0.0486 |
| Healthy Volunteers Cohort (All) | Yes (All 18 Herbs) | 106 | 48 pos / 58 neg | 0.8915 ± 0.0329 | 0.8968 ± 0.0332 | 0.7486 ± 0.0323 | 0.4462 ± 0.0759 |
| Healthy Volunteers Cohort (Excl. 4 Pure-Negative Herbs) | No (14 Herbs) | 83 | 48 pos / 35 neg | 0.8649 ± 0.0268 | 0.8788 ± 0.0244 | 0.5802 ± 0.0665 | 0.4088 ± 0.0421 |
| Case Reports Cohort (All) | Yes (18 Herbs) | 16 | 14 pos / 2 neg | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |

### Diagnostic Observations on Genre Subsets:
- In the clinical trials cohort ($N=125$), `overlap_6d` achieves a per-fold mean AUC of **0.9027**.
- When excluding the 4 pure-negative botanicals ($N=100$), `overlap_6d` per-fold AUC is **0.8779**, while `curation_2d` drops from **0.7688** to **0.6354**.
- The sanitized `design_only` representation achieves **0.4743** on clinical trials and **0.4328** with negative controls excluded.
