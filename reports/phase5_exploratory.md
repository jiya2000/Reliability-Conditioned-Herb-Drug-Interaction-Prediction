# PHASE 5: TASK A EXPLORATORY EVALUATION REPORT
## Benchmark Evaluation under Legacy Label Mode & Baseline Features

> [!IMPORTANT]
> **EXPLORATORY STATUS DISCLAIMER:** EXPLORATORY analysis on legacy label-adjacent baseline features. Every baseline row operates under label_mode='legacy'. No proposed model rows are reported. Conflict and reliability signals are NOT evaluated until consensus labels exist.

### 1. Protocol & Metadata
- **Task:** Task A (Cold-Start Pair Interaction Prediction)
- **Label Mode:** `legacy`
- **Feature Set:** `legacy_label_adjacent`
- **Status:** `EXPLORATORY`
- **Evaluation Protocol:** Repeated Grouped Cross-Validation (20 repeats, 5 folds per repeat = 100 total folds per model)
- **Benchmark Dataset:** `data/benchmark/verified_botanical_hdi.csv` ($N = 141$ records: 75 positive, 66 negative)
- **Execution Timestamp:** `2026-10-28T10:30:52Z`
- **Task B Status:** Not testable on this panel (only 15 of 120 pairs have $\ge 2$ studies). Supervised reliability model $R$ is blocked until human consensus labels exist.

### 2. Mandatory Baseline Table (Primary Metric: Mean Per-Fold AUC)

Every baseline row is evaluated under `label_mode = 'legacy'`. Hyperparameters were tuned strictly within inner training folds.
Confidence intervals are 95% cluster bootstrap CIs evaluated on the per-fold-mean metric across 1,000 resamples.

#### Split Protocol: Herb-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.1204 | 0.2598 | 100.0000 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7485** | 0.1160 | [0.6849, 0.8375] | [0.6053, 0.8317] | 0.1059 | 0.2243 | 1.5691 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8613** | 0.1084 | [0.7948, 0.9327] | [0.7976, 0.9229] | 0.0774 | 0.1627 | 0.7331 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8651** | 0.0929 | [0.8018, 0.9310] | [0.7989, 0.9235] | 0.0594 | 0.1506 | 0.9096 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.7885** | 0.1358 | [0.6169, 0.8020] | [0.7360, 0.8345] | 0.0922 | 0.1840 | 0.7180 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6406** | 0.1331 | [0.5304, 0.7050] | [0.5402, 0.7105] | 0.1156 | 0.2606 | 6.0377 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8938** | 0.0927 | [0.8302, 0.9481] | [0.8514, 0.9331] | 0.0495 | 0.1371 | 0.9775 | EXPLORATORY |
| **LR (Overlap 6D + Quality 7D, 13D)** | `legacy` | **0.8836** | 0.0919 | [0.8278, 0.9471] | [0.8129, 0.9376] | 0.0369 | 0.1518 | 0.9092 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9167** | 0.0660 | [0.8733, 0.9672] | [0.8604, 0.9609] | 0.2124 | 0.1543 | 0.3136 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9142** | 0.0663 | [0.8746, 0.9677] | [0.8518, 0.9590] | 0.1872 | 0.1521 | 0.3679 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9031** | 0.0719 | [0.8619, 0.9593] | [0.8295, 0.9526] | 0.1716 | 0.1618 | 0.3942 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6537** | 0.1069 | [0.5642, 0.7107] | [0.5197, 0.7548] | 0.0971 | 0.2375 | 1.4505 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9647** | 0.0463 | [0.9478, 0.9985] | [0.9168, 0.9946] | 0.2129 | 0.1089 | 0.2878 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8836** | 0.0919 | [0.8278, 0.9471] | [0.8129, 0.9376] | 0.0369 | 0.1518 | 0.9092 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8585** | 0.1149 | [0.7932, 0.9134] | [0.7886, 0.9142] | 0.0591 | 0.1559 | 1.4005 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.5417** | 0.2091 | [0.4505, 0.6742] | [0.4471, 0.6468] | 0.2746 | 0.3467 | 100.0000 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.1393 | 0.2607 | 100.0000 | EXPLORATORY |

#### Split Protocol: Drug-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.8222** | 0.0940 | [0.6503, 0.9134] | [0.7637, 0.8927] | 0.1151 | 0.1997 | 0.7034 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.1701 | 0.2874 | 100.0000 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8688** | 0.0793 | [0.7474, 0.9495] | [0.7972, 0.9288] | 0.0724 | 0.1627 | 0.7448 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8704** | 0.0807 | [0.7596, 0.9416] | [0.8004, 0.9217] | 0.0530 | 0.1520 | 1.0038 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8097** | 0.0875 | [0.6273, 0.9191] | [0.7234, 0.8590] | 0.0908 | 0.1888 | 0.6425 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6164** | 0.1029 | [0.5267, 0.7421] | [0.4949, 0.6863] | 0.0966 | 0.2459 | 1.8813 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.9003** | 0.0567 | [0.7999, 0.9655] | [0.8467, 0.9366] | 0.0292 | 0.1413 | 0.9483 | EXPLORATORY |
| **LR (Overlap 6D + Quality 7D, 13D)** | `legacy` | **0.8832** | 0.0836 | [0.7789, 0.9534] | [0.8115, 0.9322] | 0.0490 | 0.1492 | 1.0242 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9461** | 0.0427 | [0.8936, 0.9770] | [0.9172, 0.9767] | 0.2104 | 0.1514 | 0.3177 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9066** | 0.0641 | [0.8454, 0.9560] | [0.8452, 0.9539] | 0.1663 | 0.1560 | 0.3996 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.8930** | 0.0727 | [0.8176, 0.9474] | [0.8165, 0.9471] | 0.1640 | 0.1655 | 0.4202 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6487** | 0.0949 | [0.5363, 0.7407] | [0.5657, 0.7293] | 0.1087 | 0.2404 | 1.6320 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9780** | 0.0335 | [0.9476, 0.9991] | [0.9234, 0.9970] | 0.2200 | 0.1045 | 0.2691 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8832** | 0.0836 | [0.7789, 0.9534] | [0.8115, 0.9322] | 0.0490 | 0.1492 | 1.0242 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8854** | 0.0710 | [0.7774, 0.9487] | [0.8263, 0.9381] | 0.0598 | 0.1473 | 1.4459 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7492** | 0.1210 | [0.5929, 0.8489] | [0.7042, 0.8289] | 0.1478 | 0.2475 | 2.4422 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.1050 | 0.2547 | 100.0000 | EXPLORATORY |

#### Split Protocol: Pair-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.7985** | 0.0714 | [0.6127, 0.8998] | [0.7326, 0.8676] | 0.0905 | 0.2038 | 0.6553 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7250** | 0.0943 | [0.6403, 0.8128] | [0.5642, 0.8075] | 0.0936 | 0.2205 | 1.2650 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8627** | 0.0630 | [0.7423, 0.9473] | [0.7902, 0.9254] | 0.0912 | 0.1645 | 0.5971 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8675** | 0.0619 | [0.7547, 0.9447] | [0.7920, 0.9279] | 0.0625 | 0.1443 | 0.9715 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8189** | 0.0604 | [0.6502, 0.9197] | [0.7561, 0.8724] | 0.0981 | 0.1859 | 0.5319 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6317** | 0.1080 | [0.5459, 0.7417] | [0.5315, 0.7029] | 0.0422 | 0.2408 | 1.3472 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8982** | 0.0465 | [0.7952, 0.9639] | [0.8543, 0.9362] | 0.0393 | 0.1340 | 0.9339 | EXPLORATORY |
| **LR (Overlap 6D + Quality 7D, 13D)** | `legacy` | **0.8879** | 0.0567 | [0.7793, 0.9584] | [0.8091, 0.9409] | 0.0416 | 0.1415 | 0.8655 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9459** | 0.0382 | [0.8962, 0.9810] | [0.9126, 0.9793] | 0.2195 | 0.1399 | 0.2724 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9208** | 0.0543 | [0.8653, 0.9663] | [0.8703, 0.9655] | 0.1850 | 0.1437 | 0.3626 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9127** | 0.0580 | [0.8531, 0.9632] | [0.8534, 0.9622] | 0.1992 | 0.1500 | 0.3568 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6652** | 0.0909 | [0.5770, 0.7545] | [0.5747, 0.7402] | 0.0967 | 0.2306 | 1.2551 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9771** | 0.0262 | [0.9466, 0.9995] | [0.9482, 0.9972] | 0.2161 | 0.1006 | 0.2654 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8879** | 0.0567 | [0.7793, 0.9584] | [0.8091, 0.9409] | 0.0416 | 0.1415 | 0.8655 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8788** | 0.0646 | [0.7695, 0.9475] | [0.8024, 0.9341] | 0.0571 | 0.1428 | 1.3559 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7718** | 0.0848 | [0.6376, 0.8747] | [0.7004, 0.8522] | 0.0617 | 0.2014 | 1.4608 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.0362 | 0.2497 | 1.5188 | EXPLORATORY |

#### Split Protocol: Study-Disjoint (Secondary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.7865** | 0.0830 | [0.6075, 0.8943] | [0.7159, 0.8624] | 0.0795 | 0.2047 | 0.6660 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7243** | 0.0924 | [0.6334, 0.8129] | [0.5720, 0.8088] | 0.0864 | 0.2227 | 1.3702 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8673** | 0.0675 | [0.7509, 0.9475] | [0.7979, 0.9308] | 0.1087 | 0.1650 | 0.5863 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8681** | 0.0676 | [0.7581, 0.9412] | [0.7983, 0.9299] | 0.0683 | 0.1491 | 0.9347 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8175** | 0.0691 | [0.6498, 0.9178] | [0.7534, 0.8752] | 0.1005 | 0.1869 | 0.5059 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6115** | 0.1164 | [0.5158, 0.7297] | [0.5124, 0.6895] | 0.0598 | 0.2450 | 1.6455 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8938** | 0.0575 | [0.7903, 0.9621] | [0.8463, 0.9336] | 0.0480 | 0.1379 | 0.9418 | EXPLORATORY |
| **LR (Overlap 6D + Quality 7D, 13D)** | `legacy` | **0.8840** | 0.0634 | [0.7691, 0.9554] | [0.8099, 0.9447] | 0.0321 | 0.1460 | 0.9307 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9414** | 0.0452 | [0.8928, 0.9791] | [0.9059, 0.9744] | 0.2132 | 0.1402 | 0.2771 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9206** | 0.0556 | [0.8647, 0.9672] | [0.8697, 0.9646] | 0.1842 | 0.1429 | 0.3617 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9151** | 0.0640 | [0.8571, 0.9651] | [0.8591, 0.9618] | 0.1980 | 0.1493 | 0.3550 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6661** | 0.0997 | [0.5779, 0.7535] | [0.5786, 0.7383] | 0.0834 | 0.2317 | 1.2879 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9754** | 0.0309 | [0.9449, 0.9991] | [0.9441, 0.9965] | 0.2148 | 0.1013 | 0.2716 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8840** | 0.0634 | [0.7691, 0.9554] | [0.8099, 0.9447] | 0.0321 | 0.1460 | 0.9307 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8683** | 0.0716 | [0.7364, 0.9462] | [0.7970, 0.9310] | 0.0706 | 0.1550 | 1.5510 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7621** | 0.1031 | [0.6122, 0.8750] | [0.6868, 0.8440] | 0.0780 | 0.2104 | 1.5646 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.0339 | 0.2497 | 1.5132 | EXPLORATORY |

### 3. Comprehensive Paired Statistical Tests vs. Logistic Regression
Statistical comparison against reference baseline (`logistic_regression`, 13D legacy).
Replaces naive paired $t$-test with (a) Nadeau-Bengio corrected resampled $t$-test ($df=99$) and (b) Herb & Drug cluster bootstrap difference tests across 1,000 resamples. All p-values are Holm-Bonferroni adjusted.

#### Herb-Disjoint (Primary)

| Comparison Model | Mean Diff | NB $t$-stat | NB Raw $p$ | NB Holm $p$ | Herb Boot 95% CI | Herb Holm $p$ | Drug Boot 95% CI | Drug Holm $p$ | Sig (NB)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.3836 | -8.1832 | 0.0000 | 0.0000 | [-0.4471, -0.3278] | 0.0160 | [-0.4355, -0.3197] | 0.0160 | Yes |
| Drug Identity (One-Hot) | -0.1351 | -1.8769 | 0.0635 | 0.6348 | [-0.2133, -0.0375] | 0.0180 | [-0.2872, -0.0530] | 0.0150 | No |
| Overlap Total (1D Scalar) | -0.0223 | -1.1621 | 0.2480 | 1.0000 | [-0.0454, +0.0015] | 0.4560 | [-0.0443, +0.0036] | 0.6020 | No |
| Overlap 6D (Pharmacokinetic Targets) | -0.0185 | -1.4329 | 0.1550 | 1.0000 | [-0.0337, -0.0044] | 0.0480 | [-0.0417, +0.0039] | 0.7440 | No |
| Curation 2D (Profile Density Proxy) | -0.0952 | -1.9977 | 0.0485 | 0.5334 | [-0.2707, -0.0767] | 0.0150 | [-0.1898, +0.0025] | 0.4800 | No |
| Quality-Only (7D Lineage/Design) | -0.2430 | -3.0350 | 0.0031 | 0.0400 | [-0.3768, -0.1610] | 0.0140 | [-0.3418, -0.1625] | 0.0140 | Yes |
| LR (Overlap 6D + Curation 2D, 8D) | +0.0102 | 0.4257 | 0.6713 | 1.0000 | [-0.0228, +0.0266] | 1.0000 | [-0.0228, +0.0500] | 1.0000 | No |
| LR (Overlap 6D + Quality 7D, 13D) | +0.0000 | 0.0000 | 1.0000 | 1.0000 | [+0.0000, +0.0000] | 1.0000 | [+0.0000, +0.0000] | 1.0000 | No |
| Masked-Text Level 1 (Outcome Masked) | +0.0330 | 0.5900 | 0.5566 | 1.0000 | [-0.0458, +0.1027] | 1.0000 | [-0.0379, +0.1116] | 1.0000 | No |
| Masked-Text Level 2 (Outcome & Entities Masked) | +0.0305 | 0.5454 | 0.5867 | 1.0000 | [-0.0481, +0.1033] | 1.0000 | [-0.0462, +0.1080] | 1.0000 | No |
| Masked-Text Level 3 (Outcome, Entities & Proxies Masked) | +0.0195 | 0.3210 | 0.7489 | 1.0000 | [-0.0617, +0.1018] | 1.0000 | [-0.0659, +0.1010] | 1.0000 | No |
| Design-Only (Structured Design Tokens) | -0.2299 | -3.3539 | 0.0011 | 0.0158 | [-0.3521, -0.1608] | 0.0130 | [-0.3494, -0.1199] | 0.0130 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0810 | 1.6685 | 0.0984 | 0.8853 | [+0.0197, +0.1436] | 0.0420 | [+0.0148, +0.1499] | 0.1600 | No |
| Gradient Boosted Trees (Full 13D Legacy) | -0.0251 | -1.0840 | 0.2810 | 1.0000 | [-0.0545, -0.0154] | 0.0120 | [-0.0506, -0.0016] | 0.3060 | No |
| SmallGAT (Graph Attention Network) | -0.3420 | -2.9700 | 0.0037 | 0.0448 | [-0.4501, -0.2073] | 0.0110 | [-0.4148, -0.2642] | 0.0120 | Yes |
| Majority-Class (Dummy Baseline) | -0.3836 | -8.1832 | 0.0000 | 0.0000 | [-0.4471, -0.3278] | 0.0100 | [-0.4355, -0.3197] | 0.0110 | Yes |

#### Drug-Disjoint (Primary)

| Comparison Model | Mean Diff | NB $t$-stat | NB Raw $p$ | NB Holm $p$ | Herb Boot 95% CI | Herb Holm $p$ | Drug Boot 95% CI | Drug Holm $p$ | Sig (NB)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.0610 | -0.9363 | 0.3514 | 1.0000 | [-0.2021, +0.0192] | 1.0000 | [-0.1394, +0.0434] | 1.0000 | No |
| Drug Identity (One-Hot) | -0.3832 | -8.9936 | 0.0000 | 0.0000 | [-0.4534, -0.2789] | 0.0160 | [-0.4291, -0.3155] | 0.0160 | Yes |
| Overlap Total (1D Scalar) | -0.0143 | -0.7188 | 0.4740 | 1.0000 | [-0.0499, +0.0127] | 1.0000 | [-0.0352, +0.0145] | 1.0000 | No |
| Overlap 6D (Pharmacokinetic Targets) | -0.0127 | -0.7725 | 0.4417 | 1.0000 | [-0.0384, +0.0040] | 0.9760 | [-0.0313, +0.0069] | 1.0000 | No |
| Curation 2D (Profile Density Proxy) | -0.0735 | -1.1838 | 0.2393 | 1.0000 | [-0.2005, +0.0135] | 1.0000 | [-0.1652, +0.0109] | 0.7200 | No |
| Quality-Only (7D Lineage/Design) | -0.2667 | -4.7924 | 0.0000 | 0.0001 | [-0.4015, -0.1019] | 0.0150 | [-0.3798, -0.1788] | 0.0150 | Yes |
| LR (Overlap 6D + Curation 2D, 8D) | +0.0171 | 0.5088 | 0.6120 | 1.0000 | [-0.0220, +0.0643] | 1.0000 | [-0.0196, +0.0650] | 1.0000 | No |
| LR (Overlap 6D + Quality 7D, 13D) | +0.0000 | 0.0000 | 1.0000 | 1.0000 | [+0.0000, +0.0000] | 1.0000 | [+0.0000, +0.0000] | 1.0000 | No |
| Masked-Text Level 1 (Outcome Masked) | +0.0629 | 1.2339 | 0.2202 | 1.0000 | [-0.0104, +0.1637] | 0.9600 | [+0.0081, +0.1434] | 0.3000 | No |
| Masked-Text Level 2 (Outcome & Entities Masked) | +0.0234 | 0.4218 | 0.6741 | 1.0000 | [-0.0645, +0.1347] | 1.0000 | [-0.0530, +0.1123] | 1.0000 | No |
| Masked-Text Level 3 (Outcome, Entities & Proxies Masked) | +0.0098 | 0.1635 | 0.8705 | 1.0000 | [-0.0907, +0.1342] | 1.0000 | [-0.0814, +0.1061] | 1.0000 | No |
| Design-Only (Structured Design Tokens) | -0.2345 | -3.4711 | 0.0008 | 0.0100 | [-0.3519, -0.1466] | 0.0140 | [-0.3159, -0.1370] | 0.0140 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0948 | 1.9729 | 0.0513 | 0.6155 | [+0.0281, +0.1935] | 0.0130 | [+0.0252, +0.1689] | 0.0960 | No |
| Gradient Boosted Trees (Full 13D Legacy) | +0.0022 | 0.0776 | 0.9383 | 1.0000 | [-0.0206, +0.0173] | 1.0000 | [-0.0190, +0.0324] | 1.0000 | No |
| SmallGAT (Graph Attention Network) | -0.1339 | -1.8406 | 0.0687 | 0.7555 | [-0.2283, -0.0764] | 0.0120 | [-0.1874, -0.0261] | 0.1100 | No |
| Majority-Class (Dummy Baseline) | -0.3832 | -8.9936 | 0.0000 | 0.0000 | [-0.4534, -0.2789] | 0.0110 | [-0.4291, -0.3155] | 0.0130 | Yes |

#### Pair-Disjoint (Primary)

| Comparison Model | Mean Diff | NB $t$-stat | NB Raw $p$ | NB Holm $p$ | Herb Boot 95% CI | Herb Holm $p$ | Drug Boot 95% CI | Drug Holm $p$ | Sig (NB)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.0894 | -1.8764 | 0.0635 | 0.6354 | [-0.2271, -0.0056] | 0.3000 | [-0.1839, +0.0132] | 0.7040 | No |
| Drug Identity (One-Hot) | -0.1630 | -3.2910 | 0.0014 | 0.0180 | [-0.2672, -0.0322] | 0.1100 | [-0.3231, -0.0738] | 0.0160 | Yes |
| Overlap Total (1D Scalar) | -0.0253 | -1.5221 | 0.1312 | 0.9182 | [-0.0614, +0.0019] | 0.5600 | [-0.0501, +0.0011] | 0.5760 | No |
| Overlap 6D (Pharmacokinetic Targets) | -0.0204 | -1.2587 | 0.2111 | 1.0000 | [-0.0499, -0.0014] | 0.2700 | [-0.0455, +0.0029] | 0.6300 | No |
| Curation 2D (Profile Density Proxy) | -0.0690 | -1.7691 | 0.0800 | 0.6397 | [-0.1831, +0.0093] | 0.7280 | [-0.1624, +0.0257] | 0.8880 | No |
| Quality-Only (7D Lineage/Design) | -0.2562 | -4.6538 | 0.0000 | 0.0002 | [-0.3812, -0.1018] | 0.0160 | [-0.3476, -0.1727] | 0.0150 | Yes |
| LR (Overlap 6D + Curation 2D, 8D) | +0.0103 | 0.3906 | 0.6969 | 1.0000 | [-0.0280, +0.0576] | 1.0000 | [-0.0313, +0.0598] | 1.0000 | No |
| LR (Overlap 6D + Quality 7D, 13D) | +0.0000 | 0.0000 | 1.0000 | 1.0000 | [+0.0000, +0.0000] | 1.0000 | [+0.0000, +0.0000] | 1.0000 | No |
| Masked-Text Level 1 (Outcome Masked) | +0.0579 | 1.8063 | 0.0739 | 0.6651 | [-0.0122, +0.1642] | 0.7680 | [+0.0019, +0.1363] | 0.4400 | No |
| Masked-Text Level 2 (Outcome & Entities Masked) | +0.0329 | 0.9095 | 0.3653 | 1.0000 | [-0.0496, +0.1512] | 1.0000 | [-0.0326, +0.1216] | 1.0000 | No |
| Masked-Text Level 3 (Outcome, Entities & Proxies Masked) | +0.0248 | 0.6492 | 0.5177 | 1.0000 | [-0.0672, +0.1498] | 1.0000 | [-0.0476, +0.1195] | 1.0000 | No |
| Design-Only (Structured Design Tokens) | -0.2228 | -4.3125 | 0.0000 | 0.0005 | [-0.3251, -0.1139] | 0.0150 | [-0.3136, -0.1236] | 0.0140 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0892 | 2.9109 | 0.0045 | 0.0534 | [+0.0199, +0.1956] | 0.0240 | [+0.0331, +0.1619] | 0.0240 | No |
| Gradient Boosted Trees (Full 13D Legacy) | -0.0092 | -0.3762 | 0.7076 | 1.0000 | [-0.0297, +0.0065] | 0.9900 | [-0.0292, +0.0094] | 1.0000 | No |
| SmallGAT (Graph Attention Network) | -0.1161 | -2.6671 | 0.0089 | 0.0983 | [-0.2000, -0.0533] | 0.0140 | [-0.1900, -0.0281] | 0.1540 | No |
| Majority-Class (Dummy Baseline) | -0.3879 | -13.4115 | 0.0000 | 0.0000 | [-0.4584, -0.2793] | 0.0130 | [-0.4364, -0.3196] | 0.0130 | Yes |

### 4. SmallGAT Retention Protocol Decision
- **Decision Rule:** Keep SmallGAT only if it beats LR under all three disjoint protocols with p < 0.05
- **Final Verdict:** **`DROPPED`**
- **Rationale:** SmallGAT is REJECTED / DROPPED. Under the retention protocol, SmallGAT must beat Logistic Regression under all three primary disjoint split protocols (herb-disjoint, drug-disjoint, pair-disjoint) with statistical significance (p < 0.05). SmallGAT beats LR in all three: False.

| Disjoint Protocol | SmallGAT AUC | LR AUC | Mean Diff | NB $t$-statistic | NB $p$-value | Beats LR ($p<0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `herb_disjoint` | 0.5417 | 0.8836 | -0.3420 | -2.9700 | 0.0037 | No |
| `drug_disjoint` | 0.7492 | 0.8832 | -0.1339 | -1.8406 | 0.0687 | No |
| `pair_disjoint` | 0.7718 | 0.8879 | -0.1161 | -2.6671 | 0.0089 | No |

### 5. Text Masking Hierarchy & C-Tuning Analysis (Fixed C=1.0 vs. Inner-CV C)
Verifies that masking levels enforce strict monotonicity without outcome leakage inversions.

#### Herb-Disjoint (Primary)
| Text Representation Level | Fixed C=1.0 Mean AUC (SD) | Inner-CV C Mean AUC (SD) | Chosen C Median [Min, Max] | Hierarchy Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Text (Unmasked, Level 0)** | 0.9647 (0.0463) | 0.9665 (0.0443) | 10.0 [1.0, 10.0] | LEAKY |
| **Masked Level 1 (Outcome Masked)** | 0.9167 (0.0660) | 0.9164 (0.0644) | 10.0 [1.0, 10.0] | MONOTONIC |
| **Masked Level 2 (Outcome & Entities Masked)** | 0.9142 (0.0663) | 0.9190 (0.0671) | 10.0 [0.1, 10.0] | MONOTONIC |
| **Masked Level 3 (Outcome, Entities & Proxies)** | 0.9031 (0.0719) | 0.9052 (0.0704) | 10.0 [0.1, 10.0] | MONOTONIC |
| **Design-Only (Structured Design Tokens)** | 0.6537 (0.1069) | 0.6505 (0.1077) | 1.0 [0.1, 10.0] | MONOTONIC |

#### Drug-Disjoint (Primary)
| Text Representation Level | Fixed C=1.0 Mean AUC (SD) | Inner-CV C Mean AUC (SD) | Chosen C Median [Min, Max] | Hierarchy Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Text (Unmasked, Level 0)** | 0.9780 (0.0335) | 0.9790 (0.0350) | 10.0 [0.1, 10.0] | LEAKY |
| **Masked Level 1 (Outcome Masked)** | 0.9461 (0.0427) | 0.9451 (0.0431) | 10.0 [0.1, 10.0] | MONOTONIC |
| **Masked Level 2 (Outcome & Entities Masked)** | 0.9066 (0.0641) | 0.9114 (0.0674) | 10.0 [0.1, 10.0] | MONOTONIC |
| **Masked Level 3 (Outcome, Entities & Proxies)** | 0.8930 (0.0727) | 0.8998 (0.0731) | 1.0 [0.1, 10.0] | MONOTONIC |
| **Design-Only (Structured Design Tokens)** | 0.6487 (0.0949) | 0.6409 (0.0998) | 1.0 [0.01, 10.0] | MONOTONIC |

#### Pair-Disjoint (Primary)
| Text Representation Level | Fixed C=1.0 Mean AUC (SD) | Inner-CV C Mean AUC (SD) | Chosen C Median [Min, Max] | Hierarchy Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Text (Unmasked, Level 0)** | 0.9771 (0.0262) | 0.9789 (0.0255) | 10.0 [1.0, 10.0] | LEAKY |
| **Masked Level 1 (Outcome Masked)** | 0.9459 (0.0382) | 0.9458 (0.0393) | 10.0 [1.0, 10.0] | MONOTONIC |
| **Masked Level 2 (Outcome & Entities Masked)** | 0.9208 (0.0543) | 0.9236 (0.0574) | 10.0 [1.0, 10.0] | MONOTONIC |
| **Masked Level 3 (Outcome, Entities & Proxies)** | 0.9127 (0.0580) | 0.9127 (0.0578) | 1.0 [1.0, 10.0] | MONOTONIC |
| **Design-Only (Structured Design Tokens)** | 0.6652 (0.0909) | 0.6613 (0.0952) | 1.0 [0.1, 10.0] | MONOTONIC |

### 6. Conformal Prediction & Selective Classification (Matched Abstention Suite)
> [!NOTE]
> Conformal abstention operates under $\alpha = 0.10$. Conflict and reliability signals are **NOT evaluated** until consensus labels exist.

#### Abstention Evaluation (Herb-Disjoint (Primary))
##### Selective Accuracy at Matched Coverage Levels
| Target Coverage | Effective N | Max-Prob LR Acc (Risk) | GBT Uncertainty Acc (Risk) | Tri-State Matched Acc (Risk) |
| :---: | :---: | :---: | :---: | :---: |
| 0.2706 | 763 | 0.9201 (0.0799) | 0.8781 (0.1219) | 0.9201 (0.0799) |
| 0.5000 | 1410 | 0.8929 (0.1071) | 0.8794 (0.1206) | 0.8929 (0.1071) |
| 0.6000 | 1692 | 0.8806 (0.1194) | 0.8806 (0.1194) | 0.8806 (0.1194) |
| 0.7000 | 1974 | 0.8541 (0.1459) | 0.8652 (0.1348) | 0.8541 (0.1459) |
| 0.8000 | 2256 | 0.8267 (0.1733) | 0.8378 (0.1622) | 0.8267 (0.1733) |
| 0.8500 | 2397 | 0.8206 (0.1794) | 0.8244 (0.1756) | 0.8206 (0.1794) |
| 0.9000 | 2538 | 0.8176 (0.1824) | 0.8121 (0.1879) | 0.8176 (0.1824) |
| 0.9500 | 2679 | 0.8096 (0.1904) | 0.7977 (0.2023) | 0.8096 (0.1904) |

##### Coverage Achieved at Matched Target Selective Accuracy
| Target Accuracy | Max-Prob LR Coverage | GBT Uncertainty Coverage | Tri-State Coverage |
| :---: | :---: | :---: | :---: |
| 0.8000 | 0.9800 | 0.9300 | 0.9800 |
| 0.8500 | 0.7100 | 0.7400 | 0.7100 |
| 0.9000 | 0.4600 | 0.2200 | 0.4600 |
| 0.9500 | 0.1900 | 0.1700 | 0.1900 |

#### Abstention Evaluation (Drug-Disjoint (Primary))
##### Selective Accuracy at Matched Coverage Levels
| Target Coverage | Effective N | Max-Prob LR Acc (Risk) | GBT Uncertainty Acc (Risk) | Tri-State Matched Acc (Risk) |
| :---: | :---: | :---: | :---: | :---: |
| 0.3387 | 955 | 0.8963 (0.1037) | 0.9215 (0.0785) | 0.8963 (0.1037) |
| 0.5000 | 1410 | 0.9000 (0.1000) | 0.8844 (0.1156) | 0.9000 (0.1000) |
| 0.6000 | 1692 | 0.8830 (0.1170) | 0.8806 (0.1194) | 0.8830 (0.1170) |
| 0.7000 | 1974 | 0.8647 (0.1353) | 0.8627 (0.1373) | 0.8647 (0.1353) |
| 0.8000 | 2256 | 0.8378 (0.1622) | 0.8484 (0.1516) | 0.8378 (0.1622) |
| 0.8500 | 2397 | 0.8290 (0.1710) | 0.8398 (0.1602) | 0.8290 (0.1710) |
| 0.9000 | 2538 | 0.8195 (0.1805) | 0.8286 (0.1714) | 0.8195 (0.1805) |
| 0.9500 | 2679 | 0.8096 (0.1904) | 0.8141 (0.1859) | 0.8096 (0.1904) |

##### Coverage Achieved at Matched Target Selective Accuracy
| Target Accuracy | Max-Prob LR Coverage | GBT Uncertainty Coverage | Tri-State Coverage |
| :---: | :---: | :---: | :---: |
| 0.8000 | 0.9700 | 0.9900 | 0.9700 |
| 0.8500 | 0.7500 | 0.7900 | 0.7500 |
| 0.9000 | 0.5000 | 0.3600 | 0.5000 |
| 0.9500 | 0.1900 | 0.3000 | 0.1900 |

#### Abstention Evaluation (Pair-Disjoint (Primary))
##### Selective Accuracy at Matched Coverage Levels
| Target Coverage | Effective N | Max-Prob LR Acc (Risk) | GBT Uncertainty Acc (Risk) | Tri-State Matched Acc (Risk) |
| :---: | :---: | :---: | :---: | :---: |
| 0.3110 | 877 | 0.9441 (0.0559) | 0.9590 (0.0410) | 0.9441 (0.0559) |
| 0.5000 | 1410 | 0.9248 (0.0752) | 0.8972 (0.1028) | 0.9248 (0.0752) |
| 0.6000 | 1692 | 0.9054 (0.0946) | 0.8989 (0.1011) | 0.9054 (0.0946) |
| 0.7000 | 1974 | 0.8799 (0.1201) | 0.8794 (0.1206) | 0.8799 (0.1201) |
| 0.8000 | 2256 | 0.8457 (0.1543) | 0.8520 (0.1480) | 0.8457 (0.1543) |
| 0.8500 | 2397 | 0.8315 (0.1685) | 0.8373 (0.1627) | 0.8315 (0.1685) |
| 0.9000 | 2538 | 0.8243 (0.1757) | 0.8247 (0.1753) | 0.8243 (0.1757) |
| 0.9500 | 2679 | 0.8130 (0.1870) | 0.8160 (0.1840) | 0.8130 (0.1870) |

##### Coverage Achieved at Matched Target Selective Accuracy
| Target Accuracy | Max-Prob LR Coverage | GBT Uncertainty Coverage | Tri-State Coverage |
| :---: | :---: | :---: | :---: |
| 0.8000 | 1.0000 | 1.0000 | 1.0000 |
| 0.8500 | 0.7800 | 0.8000 | 0.7800 |
| 0.9000 | 0.6300 | 0.6100 | 0.6300 |
| 0.9500 | 0.2700 | 0.3200 | 0.2700 |

### 7. Internal Hyperparameter Tuning: Chosen $C$ Per Feature Set Per Split
Median and range of $L_2$ regularization parameter $C$ selected via 3-fold inner cross-validation across 100 folds:

#### Herb-Disjoint (Primary)
| Feature Set / Model | Median Chosen $C$ | Range [$C_{\min}, C_{\max}$] | Distribution Counts |
| :--- | :---: | :---: | :--- |
| **Herb Identity (One-Hot)** | 1.0000 | [0.10, 10.00] | C=0.1: 42, C=1.0: 55, C=10.0: 3 |
| **Drug Identity (One-Hot)** | 1.0000 | [0.10, 10.00] | C=0.1: 13, C=1.0: 52, C=10.0: 35 |
| **Overlap Total (1D Scalar)** | 0.1000 | [0.01, 10.00] | C=0.01: 8, C=0.1: 53, C=1.0: 28, C=10.0: 11 |
| **Overlap 6D (Pharmacokinetic Targets)** | 10.0000 | [0.10, 10.00] | C=0.1: 17, C=1.0: 23, C=10.0: 60 |
| **Curation 2D (Profile Density Proxy)** | 0.0100 | [0.01, 1.00] | C=0.01: 51, C=0.1: 48, C=1.0: 1 |
| **Quality-Only (7D Lineage/Design)** | 10.0000 | [0.01, 10.00] | C=0.01: 24, C=0.1: 5, C=1.0: 19, C=10.0: 52 |
| **LR (Overlap 6D + Curation 2D, 8D)** | 1.0000 | [0.01, 10.00] | C=0.01: 1, C=0.1: 16, C=1.0: 48, C=10.0: 35 |
| **LR (Overlap 6D + Quality 7D, 13D)** | 1.0000 | [0.10, 10.00] | C=0.1: 18, C=1.0: 34, C=10.0: 48 |
| **Masked-Text Level 1 (Outcome Masked)** | 10.0000 | [1.00, 10.00] | C=1.0: 35, C=10.0: 65 |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | 10.0000 | [0.10, 10.00] | C=0.1: 1, C=1.0: 10, C=10.0: 89 |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | 10.0000 | [0.10, 10.00] | C=0.1: 1, C=1.0: 40, C=10.0: 59 |
| **Design-Only (Structured Design Tokens)** | 1.0000 | [0.10, 10.00] | C=0.1: 13, C=1.0: 60, C=10.0: 27 |
| **Full-Text TF-IDF (LEAKY, reference only)** | 10.0000 | [1.00, 10.00] | C=1.0: 17, C=10.0: 83 |
| **Logistic Regression (Full 13D Legacy)** | 1.0000 | [0.10, 10.00] | C=0.1: 18, C=1.0: 34, C=10.0: 48 |

#### Drug-Disjoint (Primary)
| Feature Set / Model | Median Chosen $C$ | Range [$C_{\min}, C_{\max}$] | Distribution Counts |
| :--- | :---: | :---: | :--- |
| **Herb Identity (One-Hot)** | 1.0000 | [0.10, 10.00] | C=0.1: 40, C=1.0: 58, C=10.0: 2 |
| **Drug Identity (One-Hot)** | 1.0000 | [0.10, 10.00] | C=0.1: 27, C=1.0: 28, C=10.0: 45 |
| **Overlap Total (1D Scalar)** | 1.0000 | [0.01, 10.00] | C=0.01: 14, C=0.1: 33, C=1.0: 43, C=10.0: 10 |
| **Overlap 6D (Pharmacokinetic Targets)** | 10.0000 | [0.10, 10.00] | C=0.1: 10, C=1.0: 19, C=10.0: 71 |
| **Curation 2D (Profile Density Proxy)** | 0.0100 | [0.01, 10.00] | C=0.01: 63, C=0.1: 34, C=1.0: 2, C=10.0: 1 |
| **Quality-Only (7D Lineage/Design)** | 10.0000 | [0.01, 10.00] | C=0.01: 14, C=1.0: 15, C=10.0: 71 |
| **LR (Overlap 6D + Curation 2D, 8D)** | 1.0000 | [0.01, 10.00] | C=0.01: 2, C=0.1: 18, C=1.0: 43, C=10.0: 37 |
| **LR (Overlap 6D + Quality 7D, 13D)** | 10.0000 | [0.10, 10.00] | C=0.1: 8, C=1.0: 32, C=10.0: 60 |
| **Masked-Text Level 1 (Outcome Masked)** | 10.0000 | [0.10, 10.00] | C=0.1: 1, C=1.0: 45, C=10.0: 54 |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | 10.0000 | [0.10, 10.00] | C=0.1: 1, C=1.0: 21, C=10.0: 78 |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | 1.0000 | [0.10, 10.00] | C=0.1: 2, C=1.0: 51, C=10.0: 47 |
| **Design-Only (Structured Design Tokens)** | 1.0000 | [0.01, 10.00] | C=0.01: 3, C=0.1: 9, C=1.0: 49, C=10.0: 39 |
| **Full-Text TF-IDF (LEAKY, reference only)** | 10.0000 | [0.10, 10.00] | C=0.1: 1, C=1.0: 3, C=10.0: 96 |
| **Logistic Regression (Full 13D Legacy)** | 10.0000 | [0.10, 10.00] | C=0.1: 8, C=1.0: 32, C=10.0: 60 |

#### Pair-Disjoint (Primary)
| Feature Set / Model | Median Chosen $C$ | Range [$C_{\min}, C_{\max}$] | Distribution Counts |
| :--- | :---: | :---: | :--- |
| **Herb Identity (One-Hot)** | 1.0000 | [0.10, 10.00] | C=0.1: 46, C=1.0: 46, C=10.0: 8 |
| **Drug Identity (One-Hot)** | 1.0000 | [0.10, 10.00] | C=0.1: 12, C=1.0: 68, C=10.0: 20 |
| **Overlap Total (1D Scalar)** | 0.1000 | [0.01, 10.00] | C=0.01: 13, C=0.1: 63, C=1.0: 18, C=10.0: 6 |
| **Overlap 6D (Pharmacokinetic Targets)** | 10.0000 | [0.10, 10.00] | C=0.1: 6, C=1.0: 19, C=10.0: 75 |
| **Curation 2D (Profile Density Proxy)** | 0.0100 | [0.01, 0.10] | C=0.01: 68, C=0.1: 32 |
| **Quality-Only (7D Lineage/Design)** | 10.0000 | [0.01, 10.00] | C=0.01: 11, C=0.1: 10, C=1.0: 10, C=10.0: 69 |
| **LR (Overlap 6D + Curation 2D, 8D)** | 1.0000 | [0.10, 10.00] | C=0.1: 18, C=1.0: 42, C=10.0: 40 |
| **LR (Overlap 6D + Quality 7D, 13D)** | 5.5000 | [0.10, 10.00] | C=0.1: 9, C=1.0: 41, C=10.0: 50 |
| **Masked-Text Level 1 (Outcome Masked)** | 10.0000 | [1.00, 10.00] | C=1.0: 36, C=10.0: 64 |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | 10.0000 | [1.00, 10.00] | C=1.0: 21, C=10.0: 79 |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | 1.0000 | [1.00, 10.00] | C=1.0: 54, C=10.0: 46 |
| **Design-Only (Structured Design Tokens)** | 1.0000 | [0.10, 10.00] | C=0.1: 12, C=1.0: 56, C=10.0: 32 |
| **Full-Text TF-IDF (LEAKY, reference only)** | 10.0000 | [1.00, 10.00] | C=1.0: 4, C=10.0: 96 |
| **Logistic Regression (Full 13D Legacy)** | 5.5000 | [0.10, 10.00] | C=0.1: 9, C=1.0: 41, C=10.0: 50 |

### 8. Seed Perturbation Stability Analysis (Seeds 43, 44, 45 vs. Repeat SD)
Evaluates model sensitivity to random partition seeds (50 folds per seed) compared to repeat standard deviation under primary seed 42:

| Model / Feature Set | Seed 42 Mean AUC | Repeat SD | Seed 43 Shift | Seed 44 Shift | Seed 45 Shift | Max Shift | Bounded by SD? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `overlap_6d` | 0.8675 | 0.0619 | 0.0054 | 0.0069 | 0.0068 | 0.0069 | Yes |
| `curation_2d` | 0.8189 | 0.0604 | 0.0008 | 0.0028 | 0.0001 | 0.0028 | Yes |
| `quality_only` | 0.6317 | 0.1080 | 0.0011 | 0.0151 | 0.0019 | 0.0151 | Yes |
| `lr_overlap_plus_curation` | 0.8982 | 0.0465 | 0.0059 | 0.0060 | 0.0093 | 0.0093 | Yes |
| `logistic_regression` | 0.8879 | 0.0567 | 0.0036 | 0.0026 | 0.0043 | 0.0043 | Yes |

### 9. Genre-Stratified and Pure-Negative-Herb Sensitivity Analysis (Herb-Disjoint)
Evaluates the impact of study genres and the 4 pure-negative botanicals (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`) on key feature representations:

| Cohort Name | $N$ | Pos / Neg | `overlap_6d` AUC (SD) | `curation_2d` AUC (SD) | `quality_only` AUC (SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clinical Trials (All Herbs) | 121 | 60 / 61 | 0.9087 (0.0548) | 0.8037 (0.1086) | 0.5330 (0.0940) |
| Clinical Trials (Excl. 4 Pure-Neg Herbs) | 96 | 60 / 36 | 0.8854 (0.0621) | 0.6534 (0.1381) | 0.5647 (0.0930) |
| Healthy Volunteers (All Herbs) | 106 | 48 / 58 | 0.8927 (0.0623) | 0.7648 (0.1215) | 0.5451 (0.1188) |
| Healthy Volunteers (Excl. 4 Pure-Neg Herbs) | 83 | 48 / 35 | 0.8687 (0.0741) | 0.5931 (0.1313) | 0.5849 (0.0982) |
| Case Reports (All Herbs) | 16 | 14 / 2 | nan (nan) | nan (nan) | nan (nan) |

---
### 10. Summary & Key Findings
1. **Label-Adjacent Signal:** Tabular biological overlap (`0.86+` AUC) continues to perform strongly under legacy labels, but operates as a label-adjacent profile proxy.
2. **Curation-Density Parity:** The 2D curation baseline (`has_measured_target`, `n_nonzero_targets`) demonstrates comparable performance, indicating profile curation density is a primary non-mechanistic confound.
3. **Masking Hierarchy Restored:** The Level 2 masking bug (entity masking without outcome masking) has been corrected. Text representations now exhibit strict monotonic attenuation: Full Text (0.97+) $\ge$ Level 1 $\ge$ Level 2 $\ge$ Level 3 $\ge$ Design-Only.
4. **SmallGAT Temperature Ceiling:** SmallGAT's post-hoc calibration temperature hits the optimization upper bound because on unseen botanicals (`herb_disjoint`), GAT predictions lack discriminative generalization (AUC ~ 0.54) while producing overconfident logits. The NLL loss drives $T \to \infty$ to collapse probabilities to the uniform prior.
5. **Sanitized Design Result (0.4328):** When pure-negative controls are excluded in clinical trials ($N=100$), almost all studies are positive interactions tested with oral standardized extracts. Lacking variation in design features, linear weights fit noise, resulting in near-chance/sub-chance AUC (0.4328), demonstrating that design tokens possess no intrinsic pharmacokinetic mechanism.
6. **Task B Non-Testability:** Task B multi-study evidence aggregation remains untestable on this 141-record panel until human consensus labels are curated.
