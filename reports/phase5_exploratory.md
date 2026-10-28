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
- **Execution Timestamp:** `2026-10-28T00:30:27Z`
- **Task B Status:** Not testable on this panel (only 15 of 120 pairs have $\ge 2$ studies). Supervised reliability model $R$ is blocked until human consensus labels exist.

### 2. Mandatory Baseline Table (Primary Metric: Mean Per-Fold AUC)

Every baseline row is evaluated under `label_mode = 'legacy'`. Hyperparameters were tuned strictly within inner training folds.

#### Split Protocol: Herb-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.5000** | 0.0000 | [0.2444, 0.5018] | [0.3131, 0.4102] | 0.1204 | 0.2598 | 10.0000 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7485** | 0.1160 | [0.6315, 0.7906] | [0.5083, 0.7940] | 0.1059 | 0.2243 | 1.5691 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8613** | 0.1084 | [0.7171, 0.9327] | [0.7621, 0.9115] | 0.0774 | 0.1627 | 0.7331 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8651** | 0.0929 | [0.7401, 0.9350] | [0.7797, 0.9193] | 0.0594 | 0.1506 | 0.9096 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.7885** | 0.1358 | [0.6244, 0.9107] | [0.7322, 0.8561] | 0.0922 | 0.1840 | 0.7180 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6406** | 0.1331 | [0.4162, 0.6268] | [0.4014, 0.5881] | 0.1156 | 0.2606 | 6.0377 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9167** | 0.0660 | [0.8460, 0.9634] | [0.8403, 0.9591] | 0.2124 | 0.1543 | 0.3136 | EXPLORATORY |
| **Masked-Text Level 2 (Entity Masked)** | `legacy` | **0.9658** | 0.0462 | [0.9149, 0.9962] | [0.9102, 0.9907] | 0.1986 | 0.1026 | 0.3127 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome & Entity Masked)** | `legacy` | **0.9142** | 0.0663 | [0.8428, 0.9592] | [0.8307, 0.9560] | 0.1872 | 0.1521 | 0.3679 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6537** | 0.1069 | [0.5293, 0.7227] | [0.5055, 0.7367] | 0.0971 | 0.2375 | 1.4505 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9647** | 0.0463 | [0.9134, 0.9953] | [0.9070, 0.9916] | 0.2129 | 0.1089 | 0.2878 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8836** | 0.0919 | [0.7343, 0.9376] | [0.7837, 0.9197] | 0.0369 | 0.1518 | 0.9092 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8585** | 0.1149 | [0.7273, 0.9334] | [0.7642, 0.9183] | 0.0591 | 0.1559 | 1.4005 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.5417** | 0.2091 | [0.3232, 0.6432] | [0.3561, 0.5565] | 0.2746 | 0.3467 | 10.0000 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.2292, 0.4959] | [0.2920, 0.4017] | 0.1393 | 0.2607 | 10.0000 | EXPLORATORY |

#### Split Protocol: Drug-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.8222** | 0.0940 | [0.5721, 0.8682] | [0.7017, 0.8500] | 0.1151 | 0.1997 | 0.7034 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.5000** | 0.0000 | [0.3841, 0.4589] | [0.3597, 0.4824] | 0.1701 | 0.2874 | 10.0000 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8688** | 0.0793 | [0.7172, 0.9344] | [0.7653, 0.9122] | 0.0724 | 0.1627 | 0.7448 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8704** | 0.0807 | [0.7231, 0.9348] | [0.7742, 0.9197] | 0.0530 | 0.1520 | 1.0038 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8097** | 0.0875 | [0.6096, 0.9001] | [0.7215, 0.8572] | 0.0908 | 0.1888 | 0.6425 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6164** | 0.1029 | [0.4676, 0.6424] | [0.4609, 0.6455] | 0.0966 | 0.2459 | 1.8813 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9461** | 0.0427 | [0.8803, 0.9646] | [0.8961, 0.9685] | 0.2104 | 0.1514 | 0.3177 | EXPLORATORY |
| **Masked-Text Level 2 (Entity Masked)** | `legacy` | **0.9682** | 0.0459 | [0.9275, 0.9974] | [0.9204, 0.9942] | 0.2045 | 0.1008 | 0.3225 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome & Entity Masked)** | `legacy` | **0.9066** | 0.0641 | [0.8395, 0.9469] | [0.8351, 0.9511] | 0.1663 | 0.1560 | 0.3996 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6487** | 0.0949 | [0.5187, 0.7129] | [0.5368, 0.7135] | 0.1087 | 0.2404 | 1.6320 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9780** | 0.0335 | [0.9420, 0.9982] | [0.9432, 0.9960] | 0.2200 | 0.1045 | 0.2691 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8832** | 0.0836 | [0.7385, 0.9444] | [0.7931, 0.9280] | 0.0490 | 0.1492 | 1.0242 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8854** | 0.0710 | [0.7498, 0.9456] | [0.7955, 0.9347] | 0.0598 | 0.1473 | 1.4459 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7492** | 0.1210 | [0.4984, 0.8023] | [0.6206, 0.7753] | 0.1478 | 0.2475 | 2.4423 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.3531, 0.4411] | [0.3230, 0.4809] | 0.1050 | 0.2547 | 10.0000 | EXPLORATORY |

#### Split Protocol: Pair-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.7985** | 0.0714 | [0.5700, 0.8665] | [0.6972, 0.8467] | 0.0905 | 0.2038 | 0.6553 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7250** | 0.0943 | [0.6111, 0.7958] | [0.5365, 0.7896] | 0.0936 | 0.2205 | 1.2650 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8627** | 0.0630 | [0.7148, 0.9355] | [0.7675, 0.9130] | 0.0912 | 0.1645 | 0.5971 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8675** | 0.0619 | [0.7425, 0.9414] | [0.7862, 0.9272] | 0.0625 | 0.1443 | 0.9715 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8189** | 0.0604 | [0.6339, 0.9138] | [0.7429, 0.8649] | 0.0981 | 0.1859 | 0.5319 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6317** | 0.1080 | [0.5103, 0.6910] | [0.4972, 0.6671] | 0.0422 | 0.2408 | 1.3472 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9459** | 0.0382 | [0.8874, 0.9793] | [0.8978, 0.9774] | 0.2195 | 0.1399 | 0.2724 | EXPLORATORY |
| **Masked-Text Level 2 (Entity Masked)** | `legacy` | **0.9688** | 0.0338 | [0.9242, 0.9976] | [0.9177, 0.9931] | 0.1915 | 0.0959 | 0.3234 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome & Entity Masked)** | `legacy` | **0.9208** | 0.0543 | [0.8555, 0.9663] | [0.8529, 0.9628] | 0.1850 | 0.1437 | 0.3626 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6652** | 0.0909 | [0.5493, 0.7418] | [0.5490, 0.7340] | 0.0967 | 0.2306 | 1.2551 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9771** | 0.0262 | [0.9373, 0.9989] | [0.9372, 0.9964] | 0.2161 | 0.1006 | 0.2654 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8879** | 0.0567 | [0.7683, 0.9528] | [0.8074, 0.9366] | 0.0416 | 0.1415 | 0.8655 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8788** | 0.0646 | [0.7500, 0.9465] | [0.7974, 0.9383] | 0.0571 | 0.1428 | 1.3559 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7718** | 0.0848 | [0.6229, 0.8636] | [0.6883, 0.8407] | 0.0617 | 0.2014 | 1.4608 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.4348, 0.4951] | [0.4091, 0.5033] | 0.0362 | 0.2497 | 1.5188 | EXPLORATORY |

#### Split Protocol: Study-Disjoint (Secondary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.7865** | 0.0830 | [0.5429, 0.8652] | [0.6798, 0.8407] | 0.0795 | 0.2047 | 0.6660 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7243** | 0.0924 | [0.6001, 0.7957] | [0.5168, 0.7837] | 0.0864 | 0.2227 | 1.3702 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8673** | 0.0675 | [0.7195, 0.9374] | [0.7685, 0.9149] | 0.1087 | 0.1650 | 0.5863 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8681** | 0.0676 | [0.7376, 0.9376] | [0.7798, 0.9231] | 0.0683 | 0.1491 | 0.9347 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8175** | 0.0691 | [0.6259, 0.9134] | [0.7422, 0.8641] | 0.1005 | 0.1869 | 0.5059 | EXPLORATORY |
| **Quality-Only (7D Lineage/Design)** | `legacy` | **0.6115** | 0.1164 | [0.4599, 0.6765] | [0.4743, 0.6415] | 0.0598 | 0.2450 | 1.6455 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9414** | 0.0452 | [0.8856, 0.9782] | [0.8968, 0.9760] | 0.2132 | 0.1402 | 0.2771 | EXPLORATORY |
| **Masked-Text Level 2 (Entity Masked)** | `legacy` | **0.9677** | 0.0380 | [0.9212, 0.9976] | [0.9158, 0.9928] | 0.1918 | 0.0958 | 0.3281 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome & Entity Masked)** | `legacy` | **0.9206** | 0.0556 | [0.8562, 0.9676] | [0.8551, 0.9631] | 0.1842 | 0.1429 | 0.3617 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6661** | 0.0997 | [0.5493, 0.7419] | [0.5544, 0.7362] | 0.0834 | 0.2317 | 1.2879 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9754** | 0.0309 | [0.9345, 0.9985] | [0.9354, 0.9957] | 0.2148 | 0.1013 | 0.2716 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8840** | 0.0634 | [0.7494, 0.9499] | [0.7939, 0.9320] | 0.0321 | 0.1460 | 0.9307 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8683** | 0.0716 | [0.7083, 0.9428] | [0.7784, 0.9272] | 0.0706 | 0.1550 | 1.5510 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7621** | 0.1031 | [0.5743, 0.8654] | [0.6608, 0.8310] | 0.0780 | 0.2104 | 1.5646 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.4331, 0.4994] | [0.4192, 0.5052] | 0.0339 | 0.2497 | 1.5132 | EXPLORATORY |

### 3. Holm-Bonferroni Corrected Paired Tests vs. Logistic Regression
Paired $t$-tests computed across matched folds against the reference baseline (Logistic Regression).

#### Herb-Disjoint (Primary)

| Comparison Model | Mean Difference | $t$-statistic | Raw $p$-value | Holm-Adjusted $p$ | Significant ($lpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.3836 | -41.7263 | 0.0000 | 0.0000 | Yes |
| Drug Identity (One-Hot) | -0.1351 | -9.5701 | 0.0000 | 0.0000 | Yes |
| Overlap Total (1D Scalar) | -0.0223 | -5.9257 | 0.0000 | 0.0000 | Yes |
| Overlap 6D (Pharmacokinetic Targets) | -0.0185 | -7.3065 | 0.0000 | 0.0000 | Yes |
| Curation 2D (Profile Density Proxy) | -0.0952 | -10.1866 | 0.0000 | 0.0000 | Yes |
| Quality-Only (7D Lineage/Design) | -0.2430 | -15.4754 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 1 (Outcome Masked) | +0.0330 | 3.0083 | 0.0033 | 0.0067 | Yes |
| Masked-Text Level 2 (Entity Masked) | +0.0822 | 8.6552 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 3 (Outcome & Entity Masked) | +0.0305 | 2.7813 | 0.0065 | 0.0065 | Yes |
| Design-Only (Structured Design Tokens) | -0.2299 | -17.1017 | 0.0000 | 0.0000 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0810 | 8.5079 | 0.0000 | 0.0000 | Yes |
| Gradient Boosted Trees (Full 13D Legacy) | -0.0251 | -5.5272 | 0.0000 | 0.0000 | Yes |
| SmallGAT (Graph Attention Network) | -0.3420 | -15.1441 | 0.0000 | 0.0000 | Yes |
| Majority-Class (Dummy Baseline) | -0.3836 | -41.7263 | 0.0000 | 0.0000 | Yes |

#### Drug-Disjoint (Primary)

| Comparison Model | Mean Difference | $t$-statistic | Raw $p$-value | Holm-Adjusted $p$ | Significant ($lpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.0610 | -4.7744 | 0.0000 | 0.0000 | Yes |
| Drug Identity (One-Hot) | -0.3832 | -45.8584 | 0.0000 | 0.0000 | Yes |
| Overlap Total (1D Scalar) | -0.0143 | -3.6649 | 0.0004 | 0.0012 | Yes |
| Overlap 6D (Pharmacokinetic Targets) | -0.0127 | -3.9390 | 0.0002 | 0.0006 | Yes |
| Curation 2D (Profile Density Proxy) | -0.0735 | -6.0363 | 0.0000 | 0.0000 | Yes |
| Quality-Only (7D Lineage/Design) | -0.2667 | -24.4364 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 1 (Outcome Masked) | +0.0629 | 6.2918 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 2 (Entity Masked) | +0.0850 | 8.4734 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 3 (Outcome & Entity Masked) | +0.0234 | 2.1506 | 0.0339 | 0.0679 | No |
| Design-Only (Structured Design Tokens) | -0.2345 | -17.6990 | 0.0000 | 0.0000 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0948 | 10.0600 | 0.0000 | 0.0000 | Yes |
| Gradient Boosted Trees (Full 13D Legacy) | +0.0022 | 0.3955 | 0.6933 | 0.6933 | No |
| SmallGAT (Graph Attention Network) | -0.1339 | -9.3851 | 0.0000 | 0.0000 | Yes |
| Majority-Class (Dummy Baseline) | -0.3832 | -45.8584 | 0.0000 | 0.0000 | Yes |

#### Pair-Disjoint (Primary)

| Comparison Model | Mean Difference | $t$-statistic | Raw $p$-value | Holm-Adjusted $p$ | Significant ($lpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.0894 | -9.5680 | 0.0000 | 0.0000 | Yes |
| Drug Identity (One-Hot) | -0.1630 | -16.7808 | 0.0000 | 0.0000 | Yes |
| Overlap Total (1D Scalar) | -0.0253 | -7.7613 | 0.0000 | 0.0000 | Yes |
| Overlap 6D (Pharmacokinetic Targets) | -0.0204 | -6.4179 | 0.0000 | 0.0000 | Yes |
| Curation 2D (Profile Density Proxy) | -0.0690 | -9.0205 | 0.0000 | 0.0000 | Yes |
| Quality-Only (7D Lineage/Design) | -0.2562 | -23.7300 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 1 (Outcome Masked) | +0.0579 | 9.2106 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 2 (Entity Masked) | +0.0808 | 12.9390 | 0.0000 | 0.0000 | Yes |
| Masked-Text Level 3 (Outcome & Entity Masked) | +0.0329 | 4.6375 | 0.0000 | 0.0000 | Yes |
| Design-Only (Structured Design Tokens) | -0.2228 | -21.9895 | 0.0000 | 0.0000 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0892 | 14.8429 | 0.0000 | 0.0000 | Yes |
| Gradient Boosted Trees (Full 13D Legacy) | -0.0092 | -1.9183 | 0.0580 | 0.0580 | No |
| SmallGAT (Graph Attention Network) | -0.1161 | -13.5996 | 0.0000 | 0.0000 | Yes |
| Majority-Class (Dummy Baseline) | -0.3879 | -68.3852 | 0.0000 | 0.0000 | Yes |

### 4. SmallGAT Retention Protocol Decision
- **Decision Rule:** Keep SmallGAT only if it beats LR under all three disjoint protocols with p < 0.05
- **Final Verdict:** **`DROPPED`**
- **Rationale:** SmallGAT is REJECTED / DROPPED. Under the retention protocol, SmallGAT must beat Logistic Regression under all three primary disjoint split protocols (herb-disjoint, drug-disjoint, pair-disjoint) with statistical significance (p < 0.05). SmallGAT beats LR in all three: False.

| Disjoint Protocol | SmallGAT AUC | LR AUC | Mean Diff | $t$-statistic | $p$-value | Beats LR ($p<0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `herb_disjoint` | 0.5417 | 0.8836 | -0.3420 | -15.1441 | 0.0000 | No |
| `drug_disjoint` | 0.7492 | 0.8832 | -0.1339 | -9.3851 | 0.0000 | No |
| `pair_disjoint` | 0.7718 | 0.8879 | -0.1161 | -13.5996 | 0.0000 | No |

### 5. Conformal Prediction & Selective Classification (Abstention Suite)
> [!NOTE]
> Conformal abstention operates under $\alpha = 0.10$. Conflict and reliability signals are **NOT evaluated** until consensus labels exist.

#### Abstention Comparison (Herb-Disjoint (Primary))
| Method / Engine | Coverage ($\alpha=0.10$) | Selective Accuracy | Coverage @ 85% Acc | Coverage @ 90% Acc | Coverage @ 95% Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Max-Probability Confidence | 0.9000 | 0.8176 | 0.7071 | 0.4447 | 0.1901 |
| GBT Model Uncertainty | 0.9004 | 0.8121 | 0.7454 | 0.2234 | 0.1652 |
| Conformal Tri-State Engine | 0.2706 | 0.9201 | N/A | N/A | N/A |

#### Abstention Comparison (Drug-Disjoint (Primary))
| Method / Engine | Coverage ($\alpha=0.10$) | Selective Accuracy | Coverage @ 85% Acc | Coverage @ 90% Acc | Coverage @ 95% Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Max-Probability Confidence | 0.9000 | 0.8195 | 0.7564 | 0.5035 | 0.1950 |
| GBT Model Uncertainty | 0.9000 | 0.8286 | 0.7904 | 0.3521 | 0.2968 |
| Conformal Tri-State Engine | 0.3387 | 0.8963 | N/A | N/A | N/A |

#### Abstention Comparison (Pair-Disjoint (Primary))
| Method / Engine | Coverage ($\alpha=0.10$) | Selective Accuracy | Coverage @ 85% Acc | Coverage @ 90% Acc | Coverage @ 95% Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Max-Probability Confidence | 0.9000 | 0.8243 | 0.7819 | 0.6199 | 0.2589 |
| GBT Model Uncertainty | 0.9000 | 0.8247 | 0.8025 | 0.5557 | 0.3032 |
| Conformal Tri-State Engine | 0.3110 | 0.9441 | N/A | N/A | N/A |

### 6. Genre-Stratified and Pure-Negative-Herb Sensitivity Analysis (Herb-Disjoint)
Evaluates the impact of study genres and the 4 pure-negative botanicals (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`) on key feature representations:

| Cohort Name | $N$ | Pos / Neg | `overlap_6d` AUC (SD) | `curation_2d` AUC (SD) | `quality_only` AUC (SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clinical Trials (All Herbs) | 121 | 60 / 61 | 0.9087 (0.0548) | 0.8037 (0.1086) | 0.5330 (0.0940) |
| Clinical Trials (Excl. 4 Pure-Neg Herbs) | 96 | 60 / 36 | 0.8854 (0.0621) | 0.6534 (0.1381) | 0.5647 (0.0930) |
| Healthy Volunteers (All Herbs) | 106 | 48 / 58 | 0.8927 (0.0623) | 0.7648 (0.1215) | 0.5451 (0.1188) |
| Healthy Volunteers (Excl. 4 Pure-Neg Herbs) | 83 | 48 / 35 | 0.8687 (0.0741) | 0.5931 (0.1313) | 0.5849 (0.0982) |
| Case Reports (All Herbs) | 16 | 14 / 2 | nan (nan) | nan (nan) | nan (nan) |

---
### 7. Summary & Findings
1. **Label-Adjacent Signal:** Tabular biological overlap (`0.86+` AUC) continues to perform strongly under legacy labels, but operates as a label-adjacent profile proxy.
2. **Curation-Density Parity:** The 2D curation baseline (`has_measured_target`, `n_nonzero_targets`) demonstrates comparable performance, indicating profile curation density is a primary non-mechanistic confound.
3. **SmallGAT Rejection:** SmallGAT failed to demonstrate statistically significant superiority over Logistic Regression across all three primary disjoint split protocols ($p \ge 0.05$) and is formally dropped.
4. **Task B Non-Testability:** Task B multi-study evidence aggregation remains untestable on this 141-record panel until human consensus labels are curated.
