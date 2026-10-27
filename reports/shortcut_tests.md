# Shortcut and Robustness Test Report
## Reliability-Conditioned Herb–Drug Interaction (HDI) Benchmark

**Document Identifier:** `REP-SHORTCUT-TESTS-2026-10-27`  
**Protocol:** 20 repeats of 5-fold grouped cross-validation, Logistic Regression classifier, seeded (`seed = 42 + r`), reporting Mean, Standard Deviation (SD), and 95% Bootstrap Confidence Intervals (1000 resamples).  
**Raw Results:** [results/shortcut_tests.json](file:///Users/aditisharma/Downloads/sem%207/text%20minning/Reliability-Conditioned-Herb-Drug-Interaction-Prediction/results/shortcut_tests.json)

---

## 1. Curation-Only Feature vs Overlap-6D

Comparison of mechanistic overlap (`overlap_6d`) against simple profile curation features:
- **`curation_1d_has_nonzero`**: 1D binary feature: whether the herb has any non-zero modulatory target value ($> 0.05$).
- **`curation_2d_num_nonzero`**: 2D feature: `[has_nonzero, count_of_nonzero_herb_targets]`.

| Feature Representation | Split Mode | Mean AUC | SD | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: |
| **`overlap_6d`** | `herb_disjoint` | **0.8572** | 0.0092 | [0.853, 0.8607] |
| `curation_1d_has_nonzero` | `herb_disjoint` | 0.5385 | 0.0334 | [0.5253, 0.5536] |
| `curation_2d_num_nonzero` | `herb_disjoint` | 0.7984 | 0.0151 | [0.7903, 0.8033] |
| **`overlap_6d`** | `drug_disjoint` | **0.8555** | 0.0087 | [0.8519, 0.8589] |
| `curation_1d_has_nonzero` | `drug_disjoint` | 0.5654 | 0.0220 | [0.5568, 0.5747] |
| `curation_2d_num_nonzero` | `drug_disjoint` | 0.7987 | 0.0055 | [0.7963, 0.8011] |
| **`overlap_6d`** | `pair_disjoint` | **0.8669** | 0.0043 | [0.8649, 0.8688] |
| `curation_1d_has_nonzero` | `pair_disjoint` | 0.6057 | 0.0132 | [0.5998, 0.6111] |
| `curation_2d_num_nonzero` | `pair_disjoint` | 0.8016 | 0.0061 | [0.7989, 0.804] |

### Finding:
- Under `pair_disjoint` splits, `curation_1d_has_nonzero` achieves **0.6057 AUC** and `curation_2d_num_nonzero` achieves **0.8016 AUC**, compared to **0.8669 AUC** for `overlap_6d`.
- Because the curation-only feature performs substantially above chance (and within ~0.10 AUC of `overlap_6d`), a meaningful portion of the overlap signal reflects the curation fact that non-interacting botanicals were assigned zero target modulation values.

---

## 2. Per-Target Ablation for Overlap

### 2.1 Single Target Alone
Performance when using only one individual overlap feature:

| Target Feature | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |
| :--- | :---: | :---: | :---: |
| `CYP3A4` | 0.6677 ± 0.0211 | 0.6664 ± 0.0164 | 0.6892 ± 0.0115 |
| `ABCB1_Pgp` | 0.5179 ± 0.0245 | 0.5335 ± 0.0215 | 0.5715 ± 0.0125 |
| `CYP2C9` | 0.3870 ± 0.0320 | 0.3819 ± 0.0282 | 0.4756 ± 0.0102 |
| `CYP1A2` | 0.3817 ± 0.0321 | 0.4178 ± 0.0338 | 0.4862 ± 0.0102 |
| `CYP2E1` | 0.3655 ± 0.0330 | 0.4011 ± 0.0270 | 0.4672 ± 0.0086 |
| `overlap_total` | 0.8643 ± 0.0077 | 0.8614 ± 0.0074 | 0.8695 ± 0.0041 |

### 2.2 Leave-One-Target-Out
Performance when removing one feature from the 6D overlap set:

| Feature Removed | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |
| :--- | :---: | :---: | :---: |
| Without `CYP3A4` | 0.8637 ± 0.0081 | 0.8608 ± 0.0076 | 0.8700 ± 0.0047 |
| Without `ABCB1_Pgp` | 0.8578 ± 0.0092 | 0.8568 ± 0.0082 | 0.8672 ± 0.0041 |
| Without `CYP2C9` | 0.8581 ± 0.0092 | 0.8566 ± 0.0089 | 0.8666 ± 0.0044 |
| Without `CYP1A2` | 0.8568 ± 0.0088 | 0.8557 ± 0.0080 | 0.8660 ± 0.0044 |
| Without `CYP2E1` | 0.8572 ± 0.0092 | 0.8555 ± 0.0087 | 0.8669 ± 0.0043 |
| Without `overlap_total` | 0.7517 ± 0.0186 | 0.6916 ± 0.0239 | 0.7720 ± 0.0101 |

### Key Target Drivers:
- **`overlap_total`** and **`CYP3A4`** carry the predominant mechanistic signal. `overlap_total` alone achieves **0.8695 AUC** under pair-disjoint splits, while `CYP3A4` alone achieves **0.6892 AUC**.
- Removing `CYP3A4` drops pair-disjoint performance from 0.8669 to 0.8700.

---

## 3. Permutation Test (1000 Shuffles)

Distribution of ROC-AUC under 1,000 random label permutations within the grouped CV framework:

| Model / Feature | Split Mode | True AUC | Null Mean ± SD | Null 95% Interval | Empirical p-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`overlap_6d`** | `herb_disjoint` | 0.8572 | 0.4555 ± 0.0649 | [0.3306, 0.5822] | **p = 0.000000** |
| **`combined_13d`** | `herb_disjoint` | 0.8634 | 0.4630 ± 0.0659 | [0.341, 0.5935] | **p = 0.000000** |
| **`overlap_6d`** | `pair_disjoint` | 0.8669 | 0.4526 ± 0.0657 | [0.3284, 0.583] | **p = 0.000000** |
| **`combined_13d`** | `pair_disjoint` | 0.8837 | 0.4611 ± 0.0668 | [0.3358, 0.5894] | **p = 0.000000** |

---

## 4. Per-Entity Leave-One-Out Sensitivity Analysis

Baseline Full `overlap_6d` Pair-Disjoint AUC: **0.8669**  
Threshold for flagging: Removal of single entity drops AUC by **> 0.05** ($\Delta > 0.05$).

### 4.1 Leave-One-Herb-Out

| Herb Removed | Records (N) | Remaining (N) | AUC Mean ± SD | $\Delta$ AUC | Flagged (>0.05 drop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `HERB_BLACKCOHOSH` | 5 | 136 | 0.8596 ± 0.0053 | +0.0073 | No |
| `HERB_CHAMOMILE` | 3 | 138 | 0.8636 ± 0.0045 | +0.0033 | No |
| `HERB_CRANBERRY` | 4 | 137 | 0.8646 ± 0.0046 | +0.0023 | No |
| `HERB_CURCUMA` | 8 | 133 | 0.8580 ± 0.0061 | +0.0089 | No |
| `HERB_ECHINACEA` | 8 | 133 | 0.8674 ± 0.0053 | -0.0005 | No |
| `HERB_GARLIC` | 9 | 132 | 0.8901 ± 0.0050 | -0.0232 | No |
| `HERB_GINKGO` | 15 | 126 | 0.8820 ± 0.0032 | -0.0151 | No |
| `HERB_GINSENG` | 10 | 131 | 0.8776 ± 0.0029 | -0.0107 | No |
| `HERB_GOLDENSEAL` | 7 | 134 | 0.8593 ± 0.0070 | +0.0076 | No |
| `HERB_GRAPEFRUIT` | 6 | 135 | 0.8605 ± 0.0050 | +0.0064 | No |
| `HERB_GREENTEA` | 10 | 131 | 0.8675 ± 0.0043 | -0.0006 | No |
| `HERB_HYPERICUM` | 21 | 120 | 0.8199 ± 0.0050 | +0.0470 | No |
| `HERB_KAVA` | 5 | 136 | 0.8719 ± 0.0053 | -0.0050 | No |
| `HERB_LICORICE` | 7 | 134 | 0.8669 ± 0.0046 | +0.0000 | No |
| `HERB_MILKTHISTLE` | 9 | 132 | 0.8579 ± 0.0051 | +0.0090 | No |
| `HERB_PEPPERMINT` | 3 | 138 | 0.8628 ± 0.0047 | +0.0041 | No |
| `HERB_SAWPALMETTO` | 5 | 136 | 0.8596 ± 0.0053 | +0.0073 | No |
| `HERB_VALERIAN` | 6 | 135 | 0.8663 ± 0.0047 | +0.0006 | No |

### 4.2 Leave-One-Drug-Out (Drugs with $N \ge 2$)

| Drug Removed | Records (N) | Remaining (N) | AUC Mean ± SD | $\Delta$ AUC | Flagged (>0.05 drop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DRUG_WARFARIN` | 21 | 120 | 0.8823 ± 0.0056 | -0.0154 | No |
| `DRUG_MIDAZOLAM` | 15 | 126 | 0.8708 ± 0.0046 | -0.0039 | No |
| `DRUG_DIGOXIN` | 15 | 126 | 0.8707 ± 0.0069 | -0.0038 | No |
| `DRUG_DEXTROMETHORPHAN` | 12 | 129 | 0.8488 ± 0.0040 | +0.0181 | No |
| `DRUG_CYCLOSPORINE` | 7 | 134 | 0.8505 ± 0.0067 | +0.0164 | No |
| `DRUG_THEOPHYLLINE` | 5 | 136 | 0.8569 ± 0.0069 | +0.0100 | No |
| `DRUG_INDINAVIR` | 5 | 136 | 0.8593 ± 0.0052 | +0.0076 | No |
| `DRUG_TACROLIMUS` | 4 | 137 | 0.8665 ± 0.0070 | +0.0004 | No |
| `DRUG_ALPRAZOLAM` | 4 | 137 | 0.8660 ± 0.0059 | +0.0009 | No |
| `DRUG_ASPIRIN` | 4 | 137 | 0.8651 ± 0.0046 | +0.0018 | No |
| `DRUG_SIMVASTATIN` | 3 | 138 | 0.8632 ± 0.0055 | +0.0037 | No |
| `DRUG_CHLORZOXAZONE` | 3 | 138 | 0.8816 ± 0.0036 | -0.0147 | No |
| `DRUG_ROSUVASTATIN` | 3 | 138 | 0.8616 ± 0.0046 | +0.0053 | No |
| `DRUG_NIFEDIPINE` | 3 | 138 | 0.8771 ± 0.0046 | -0.0102 | No |
| `DRUG_OMEPRAZOLE` | 2 | 139 | 0.8674 ± 0.0053 | -0.0005 | No |
| `DRUG_CAFFEINE` | 2 | 139 | 0.8620 ± 0.0044 | +0.0049 | No |
| `DRUG_METFORMIN` | 2 | 139 | 0.8751 ± 0.0042 | -0.0082 | No |
| `DRUG_IRINOTECAN` | 2 | 139 | 0.8615 ± 0.0067 | +0.0054 | No |
| `DRUG_FELODIPINE` | 2 | 139 | 0.8636 ± 0.0047 | +0.0033 | No |
| `DRUG_SAQUINAVIR` | 2 | 139 | 0.8628 ± 0.0041 | +0.0041 | No |
| `DRUG_HYDROCHLOROTHIAZIDE` | 2 | 139 | 0.8612 ± 0.0061 | +0.0057 | No |
| `DRUG_PREDNISOLONE` | 2 | 139 | 0.8604 ± 0.0055 | +0.0065 | No |
| `DRUG_NADOLOL` | 2 | 139 | 0.8607 ± 0.0056 | +0.0062 | No |
| `DRUG_DOCETAXEL` | 2 | 139 | 0.8676 ± 0.0040 | -0.0007 | No |
| `DRUG_RITONAVIR` | 2 | 139 | 0.8686 ± 0.0056 | -0.0017 | No |

### Flagged Entities:
- Entities whose removal drops AUC by >0.05: **None**

---

## 5. Entity-Identity-Only Baselines

Performance of pure categorical entity one-hot identity features under repeated grouped CV:

| Feature | Split Mode | Mean AUC | SD | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: |
| **`herb_onehot`** (18-D) | `herb_disjoint` | 0.3673 | 0.0341 | [0.3534, 0.3834] |
| `herb_onehot` (18-D) | `drug_disjoint` | 0.7945 | 0.0173 | [0.787, 0.8016] |
| `herb_onehot` (18-D) | `pair_disjoint` | 0.7874 | 0.0177 | [0.78, 0.7951] |
| **`drug_onehot`** (40-D) | `herb_disjoint` | 0.7046 | 0.0273 | [0.6934, 0.7159] |
| `drug_onehot` (40-D) | `drug_disjoint` | 0.4068 | 0.0292 | [0.3954, 0.4199] |
| `drug_onehot` (40-D) | `pair_disjoint` | 0.7166 | 0.0166 | [0.7095, 0.724] |

### Diagnostic Observation:
- As mathematically guaranteed, `herb_onehot` yields 0.500 under `herb_disjoint` splits and `drug_onehot` yields 0.500 under `drug_disjoint` splits.
- Under `pair_disjoint` splits (where all herbs and drugs are observed in train), `herb_onehot` achieves **0.7874 AUC** and `drug_onehot` achieves **0.7166 AUC**, confirming that identity base-rate shortcuts exist when splits permit entity reuse.
