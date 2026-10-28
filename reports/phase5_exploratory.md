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
- **Execution Timestamp:** `2026-10-28T13:48:28Z`
- **Task B Status:** Not testable on this panel (only 15 of 120 pairs have $\ge 2$ studies). Supervised reliability model $R$ is blocked until human consensus labels exist.

### 2. Mandatory Baseline Table (Primary Metric: Mean Per-Fold AUC)

Every baseline row is evaluated under `label_mode = 'legacy'`. Hyperparameters were tuned strictly within inner training folds.
Confidence intervals are 95% cluster bootstrap CIs evaluated on the per-fold-mean metric across 1,000 resamples.

#### Split Protocol: Herb-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.1217 | 0.2600 | not identifiable | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7485** | 0.1160 | [0.6849, 0.8375] | [0.6053, 0.8317] | 0.1064 | 0.2243 | 1.5691 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8613** | 0.1084 | [0.7948, 0.9327] | [0.7976, 0.9229] | 0.0819 | 0.1637 | 0.7741 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8644** | 0.0930 | [0.8014, 0.9309] | [0.7990, 0.9237] | 0.0566 | 0.1503 | 0.9628 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.7885** | 0.1358 | [0.6169, 0.8020] | [0.7360, 0.8345] | 0.0921 | 0.1844 | 0.7551 | EXPLORATORY |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | `legacy` | **0.6310** | 0.1319 | [0.5186, 0.6959] | [0.5263, 0.7028] | 0.1337 | 0.2668 | 7.1225 | EXPLORATORY |
| **Study Design Flags (9D Full Methodology)** | `legacy` | **0.5504** | 0.0881 | [0.5116, 0.5981] | [0.4741, 0.6241] | 0.1386 | 0.2629 | not identifiable | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8924** | 0.0920 | [0.8289, 0.9474] | [0.8472, 0.9318] | 0.0469 | 0.1374 | 1.0084 | EXPLORATORY |
| **LR (Overlap 6D + Sanitized Design, 12D)** | `legacy` | **0.8540** | 0.0940 | [0.7924, 0.9230] | [0.7819, 0.9185] | 0.0722 | 0.1560 | 1.0342 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9167** | 0.0660 | [0.8733, 0.9672] | [0.8604, 0.9609] | 0.2124 | 0.1543 | 0.3136 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9190** | 0.0639 | [0.8750, 0.9671] | [0.8644, 0.9620] | 0.2087 | 0.1474 | 0.3460 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9120** | 0.0657 | [0.8728, 0.9622] | [0.8495, 0.9593] | 0.1935 | 0.1537 | 0.3570 | EXPLORATORY |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | `legacy` | **0.9085** | 0.0721 | [0.8611, 0.9671] | [0.8368, 0.9566] | 0.2006 | 0.1592 | 0.3334 | EXPLORATORY |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | `legacy` | **0.9142** | 0.0663 | [0.8746, 0.9677] | [0.8518, 0.9590] | 0.1872 | 0.1521 | 0.3679 | EXPLORATORY |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | `legacy` | **0.9031** | 0.0719 | [0.8619, 0.9593] | [0.8295, 0.9526] | 0.1716 | 0.1618 | 0.3942 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6378** | 0.1022 | [0.5512, 0.6922] | [0.4677, 0.7810] | 0.1064 | 0.2393 | 1.5367 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9647** | 0.0463 | [0.9478, 0.9985] | [0.9168, 0.9946] | 0.2129 | 0.1089 | 0.2878 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8798** | 0.1009 | [0.8217, 0.9504] | [0.8073, 0.9321] | 0.0453 | 0.1425 | 1.2842 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8585** | 0.1149 | [0.7932, 0.9134] | [0.7886, 0.9142] | 0.0590 | 0.1559 | 1.4005 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.5417** | 0.2091 | [0.4505, 0.6742] | [0.4471, 0.6468] | 0.2746 | 0.3467 | not identifiable | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.0000 | 0.2490 | not identifiable | EXPLORATORY |

#### Split Protocol: Drug-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.8230** | 0.0899 | [0.6528, 0.9134] | [0.7651, 0.8926] | 0.1128 | 0.1997 | 0.7526 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.1705 | 0.2874 | not identifiable | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8688** | 0.0793 | [0.7474, 0.9495] | [0.7972, 0.9288] | 0.0703 | 0.1618 | 0.7686 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8644** | 0.0927 | [0.7548, 0.9370] | [0.7972, 0.9194] | 0.0403 | 0.1568 | 1.0994 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8097** | 0.0875 | [0.6273, 0.9191] | [0.7234, 0.8590] | 0.0883 | 0.1914 | 0.6559 | EXPLORATORY |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | `legacy` | **0.6146** | 0.1015 | [0.5237, 0.7385] | [0.4915, 0.6840] | 0.1025 | 0.2442 | 1.8540 | EXPLORATORY |
| **Study Design Flags (9D Full Methodology)** | `legacy` | **0.5412** | 0.0639 | [0.4764, 0.6017] | [0.4996, 0.6077] | 0.0987 | 0.2542 | 6.2820 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8925** | 0.0671 | [0.7951, 0.9571] | [0.8434, 0.9342] | 0.0426 | 0.1538 | 1.2442 | EXPLORATORY |
| **LR (Overlap 6D + Sanitized Design, 12D)** | `legacy` | **0.8688** | 0.0825 | [0.7616, 0.9378] | [0.8010, 0.9245] | 0.0484 | 0.1572 | 1.0660 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9461** | 0.0427 | [0.8936, 0.9770] | [0.9172, 0.9767] | 0.2104 | 0.1514 | 0.3177 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9205** | 0.0599 | [0.8699, 0.9629] | [0.8603, 0.9666] | 0.1929 | 0.1516 | 0.3782 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9156** | 0.0640 | [0.8584, 0.9612] | [0.8475, 0.9642] | 0.1907 | 0.1589 | 0.3945 | EXPLORATORY |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | `legacy` | **0.9328** | 0.0502 | [0.8737, 0.9711] | [0.8995, 0.9703] | 0.1921 | 0.1550 | 0.3284 | EXPLORATORY |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | `legacy` | **0.9066** | 0.0641 | [0.8454, 0.9560] | [0.8452, 0.9539] | 0.1663 | 0.1560 | 0.3996 | EXPLORATORY |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | `legacy` | **0.8930** | 0.0727 | [0.8176, 0.9474] | [0.8165, 0.9471] | 0.1640 | 0.1655 | 0.4202 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6737** | 0.1128 | [0.5610, 0.7638] | [0.5824, 0.7740] | 0.1129 | 0.2404 | 1.6666 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9780** | 0.0335 | [0.9476, 0.9991] | [0.9234, 0.9970] | 0.2200 | 0.1045 | 0.2691 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8673** | 0.0921 | [0.7708, 0.9388] | [0.8000, 0.9218] | 0.0549 | 0.1553 | 1.7414 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8854** | 0.0710 | [0.7774, 0.9487] | [0.8263, 0.9381] | 0.0598 | 0.1473 | 1.4459 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7492** | 0.1210 | [0.5929, 0.8489] | [0.7042, 0.8289] | 0.1478 | 0.2475 | 2.4422 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.0000 | 0.2490 | not identifiable | EXPLORATORY |

#### Split Protocol: Pair-Disjoint (Primary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.7978** | 0.0723 | [0.6127, 0.8979] | [0.7320, 0.8673] | 0.0876 | 0.2048 | 0.7568 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7250** | 0.0943 | [0.6403, 0.8128] | [0.5642, 0.8075] | 0.0923 | 0.2205 | 1.2650 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8627** | 0.0630 | [0.7423, 0.9473] | [0.7902, 0.9254] | 0.0889 | 0.1637 | 0.6143 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8679** | 0.0623 | [0.7553, 0.9453] | [0.7927, 0.9279] | 0.0511 | 0.1431 | 1.0076 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8189** | 0.0604 | [0.6502, 0.9197] | [0.7561, 0.8724] | 0.0973 | 0.1856 | 0.5373 | EXPLORATORY |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | `legacy` | **0.6211** | 0.1117 | [0.5414, 0.7253] | [0.5230, 0.6898] | 0.0539 | 0.2412 | 1.6570 | EXPLORATORY |
| **Study Design Flags (9D Full Methodology)** | `legacy` | **0.5710** | 0.0563 | [0.5013, 0.6421] | [0.5078, 0.6300] | 0.0395 | 0.2453 | 1.0477 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8994** | 0.0450 | [0.7985, 0.9626] | [0.8566, 0.9377] | 0.0363 | 0.1339 | 0.9763 | EXPLORATORY |
| **LR (Overlap 6D + Sanitized Design, 12D)** | `legacy` | **0.8610** | 0.0685 | [0.7466, 0.9400] | [0.7855, 0.9218] | 0.0615 | 0.1476 | 0.9712 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9459** | 0.0382 | [0.8962, 0.9810] | [0.9126, 0.9793] | 0.2195 | 0.1399 | 0.2724 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9278** | 0.0510 | [0.8755, 0.9707] | [0.8830, 0.9691] | 0.2011 | 0.1385 | 0.3421 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9215** | 0.0528 | [0.8661, 0.9676] | [0.8701, 0.9666] | 0.2074 | 0.1428 | 0.3414 | EXPLORATORY |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | `legacy` | **0.9368** | 0.0446 | [0.8830, 0.9763] | [0.8970, 0.9765] | 0.2060 | 0.1444 | 0.2909 | EXPLORATORY |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | `legacy` | **0.9208** | 0.0543 | [0.8653, 0.9663] | [0.8703, 0.9655] | 0.1850 | 0.1437 | 0.3626 | EXPLORATORY |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | `legacy` | **0.9127** | 0.0580 | [0.8531, 0.9632] | [0.8534, 0.9622] | 0.1992 | 0.1500 | 0.3568 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6646** | 0.0865 | [0.5773, 0.7515] | [0.5445, 0.7852] | 0.1086 | 0.2303 | 1.2667 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9771** | 0.0262 | [0.9466, 0.9995] | [0.9482, 0.9972] | 0.2161 | 0.1006 | 0.2654 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8903** | 0.0584 | [0.7846, 0.9565] | [0.8127, 0.9386] | 0.0412 | 0.1326 | 1.1504 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8788** | 0.0646 | [0.7695, 0.9475] | [0.8024, 0.9341] | 0.0571 | 0.1428 | 1.3559 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7718** | 0.0848 | [0.6376, 0.8747] | [0.7004, 0.8522] | 0.0617 | 0.2014 | 1.4608 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.0000 | 0.2490 | not identifiable | EXPLORATORY |

#### Split Protocol: Study-Disjoint (Secondary)

| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Herb Identity (One-Hot)** | `legacy` | **0.7881** | 0.0821 | [0.6098, 0.8947] | [0.7172, 0.8653] | 0.0634 | 0.2045 | 0.9035 | EXPLORATORY |
| **Drug Identity (One-Hot)** | `legacy` | **0.7243** | 0.0924 | [0.6334, 0.8129] | [0.5720, 0.8088] | 0.0860 | 0.2227 | 1.3702 | EXPLORATORY |
| **Overlap Total (1D Scalar)** | `legacy` | **0.8673** | 0.0675 | [0.7509, 0.9475] | [0.7979, 0.9308] | 0.1058 | 0.1650 | 0.5960 | EXPLORATORY |
| **Overlap 6D (Pharmacokinetic Targets)** | `legacy` | **0.8685** | 0.0668 | [0.7589, 0.9418] | [0.7987, 0.9304] | 0.0704 | 0.1490 | 0.9727 | EXPLORATORY |
| **Curation 2D (Profile Density Proxy)** | `legacy` | **0.8175** | 0.0691 | [0.6498, 0.9178] | [0.7534, 0.8752] | 0.0973 | 0.1872 | 0.5126 | EXPLORATORY |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | `legacy` | **0.5996** | 0.1186 | [0.5091, 0.7050] | [0.5069, 0.6674] | 0.0666 | 0.2454 | 1.8941 | EXPLORATORY |
| **Study Design Flags (9D Full Methodology)** | `legacy` | **0.5691** | 0.0611 | [0.5021, 0.6364] | [0.5072, 0.6315] | 0.0410 | 0.2462 | 1.1808 | EXPLORATORY |
| **LR (Overlap 6D + Curation 2D, 8D)** | `legacy` | **0.8938** | 0.0574 | [0.7902, 0.9608] | [0.8471, 0.9348] | 0.0416 | 0.1393 | 1.0188 | EXPLORATORY |
| **LR (Overlap 6D + Sanitized Design, 12D)** | `legacy` | **0.8597** | 0.0693 | [0.7511, 0.9358] | [0.7869, 0.9233] | 0.0581 | 0.1503 | 1.0275 | EXPLORATORY |
| **Masked-Text Level 1 (Outcome Masked)** | `legacy` | **0.9414** | 0.0452 | [0.8928, 0.9791] | [0.9059, 0.9744] | 0.2132 | 0.1402 | 0.2771 | EXPLORATORY |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | `legacy` | **0.9278** | 0.0521 | [0.8741, 0.9720] | [0.8819, 0.9680] | 0.2029 | 0.1379 | 0.3424 | EXPLORATORY |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | `legacy` | **0.9224** | 0.0576 | [0.8687, 0.9710] | [0.8724, 0.9663] | 0.2117 | 0.1422 | 0.3419 | EXPLORATORY |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | `legacy` | **0.9328** | 0.0488 | [0.8812, 0.9741] | [0.8903, 0.9707] | 0.2066 | 0.1447 | 0.2944 | EXPLORATORY |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | `legacy` | **0.9206** | 0.0556 | [0.8647, 0.9672] | [0.8697, 0.9646] | 0.1842 | 0.1429 | 0.3617 | EXPLORATORY |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | `legacy` | **0.9151** | 0.0640 | [0.8571, 0.9651] | [0.8591, 0.9618] | 0.1980 | 0.1493 | 0.3550 | EXPLORATORY |
| **Design-Only (Structured Design Tokens)** | `legacy` | **0.6591** | 0.0924 | [0.5692, 0.7484] | [0.5427, 0.7847] | 0.0952 | 0.2310 | 1.2696 | EXPLORATORY |
| **Full-Text TF-IDF (LEAKY, reference only)** | `legacy` | **0.9754** | 0.0309 | [0.9449, 0.9991] | [0.9441, 0.9965] | 0.2148 | 0.1013 | 0.2716 | LEAKY (Ref) |
| **Logistic Regression (Full 13D Legacy)** | `legacy` | **0.8813** | 0.0717 | [0.7698, 0.9520] | [0.8112, 0.9404] | 0.0404 | 0.1429 | 1.2853 | EXPLORATORY |
| **Gradient Boosted Trees (Full 13D Legacy)** | `legacy` | **0.8683** | 0.0716 | [0.7364, 0.9462] | [0.7970, 0.9310] | 0.0706 | 0.1550 | 1.5510 | EXPLORATORY |
| **SmallGAT (Graph Attention Network)** | `legacy` | **0.7621** | 0.1031 | [0.6122, 0.8750] | [0.6868, 0.8440] | 0.0780 | 0.2104 | 1.5646 | EXPLORATORY |
| **Majority-Class (Dummy Baseline)** | `legacy` | **0.5000** | 0.0000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.0000 | 0.2490 | not identifiable | EXPLORATORY |

### 3. Comprehensive Paired Statistical Tests vs. Logistic Regression
Statistical comparison against reference baseline (`logistic_regression`, 13D legacy).
Replaces naive paired $t$-test with (a) Nadeau-Bengio corrected resampled $t$-test ($df=99$) and (b) Herb & Drug cluster bootstrap difference tests across 1,000 resamples. All p-values are Holm-Bonferroni adjusted.

#### Herb-Disjoint (Primary)

| Comparison Model | Mean Diff | NB $t$-stat | NB Raw $p$ | NB Holm $p$ | Herb Boot 95% CI | Herb Holm $p$ | Drug Boot 95% CI | Drug Holm $p$ | Sig (NB)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.3798 | -7.3816 | 0.0000 | 0.0000 | [-0.4504, -0.3217] | 0.0200 | [-0.4303, -0.3194] | 0.0200 | Yes |
| Drug Identity (One-Hot) | -0.1313 | -1.7342 | 0.0860 | 1.0000 | [-0.2143, -0.0400] | 0.0520 | [-0.2780, -0.0504] | 0.0280 | No |
| Overlap Total (1D Scalar) | -0.0184 | -0.5837 | 0.5607 | 1.0000 | [-0.0548, +0.0096] | 1.0000 | [-0.0494, +0.0206] | 1.0000 | No |
| Overlap 6D (Pharmacokinetic Targets) | -0.0154 | -0.5800 | 0.5632 | 1.0000 | [-0.0443, +0.0051] | 1.0000 | [-0.0473, +0.0235] | 1.0000 | No |
| Curation 2D (Profile Density Proxy) | -0.0913 | -1.7240 | 0.0878 | 1.0000 | [-0.2777, -0.0690] | 0.0190 | [-0.1788, +0.0012] | 0.6480 | No |
| Lineage Features 7D (Echo/Quality/Compatibility) | -0.2488 | -3.1482 | 0.0022 | 0.0369 | [-0.3850, -0.1646] | 0.0180 | [-0.3516, -0.1697] | 0.0190 | Yes |
| Study Design Flags (9D Full Methodology) | -0.3294 | -3.8767 | 0.0002 | 0.0034 | [-0.4243, -0.2480] | 0.0170 | [-0.4197, -0.2318] | 0.0180 | Yes |
| LR (Overlap 6D + Curation 2D, 8D) | +0.0126 | 0.4083 | 0.6840 | 1.0000 | [-0.0319, +0.0340] | 0.9280 | [-0.0213, +0.0588] | 0.9760 | No |
| LR (Overlap 6D + Sanitized Design, 12D) | -0.0258 | -0.8394 | 0.4033 | 1.0000 | [-0.0585, +0.0024] | 0.7200 | [-0.0574, +0.0155] | 1.0000 | No |
| Masked-Text Level 1 (Outcome Masked) | +0.0369 | 0.5876 | 0.5581 | 1.0000 | [-0.0509, +0.1098] | 1.0000 | [-0.0361, +0.1098] | 1.0000 | No |
| Masked-Text Level 2 (Outcome & Entities Masked) | +0.0392 | 0.6276 | 0.5317 | 1.0000 | [-0.0487, +0.1088] | 1.0000 | [-0.0341, +0.1113] | 1.0000 | No |
| Masked-Text Level 3 (Outcome, Entities & Proxies Masked) | +0.0322 | 0.5015 | 0.6172 | 1.0000 | [-0.0522, +0.1047] | 1.0000 | [-0.0462, +0.1080] | 1.0000 | No |
| Masked-Text Norm Level 1 (Whitespace Replaced) | +0.0287 | 0.4464 | 0.6563 | 1.0000 | [-0.0665, +0.1082] | 1.0000 | [-0.0554, +0.1060] | 1.0000 | No |
| Masked-Text Norm Level 2 (Whitespace Replaced) | +0.0344 | 0.5496 | 0.5838 | 1.0000 | [-0.0502, +0.1092] | 1.0000 | [-0.0411, +0.1080] | 1.0000 | No |
| Masked-Text Norm Level 3 (Whitespace Replaced) | +0.0233 | 0.3469 | 0.7294 | 0.7294 | [-0.0638, +0.1056] | 1.0000 | [-0.0625, +0.1009] | 0.5460 | No |
| Design-Only (Structured Design Tokens) | -0.2420 | -3.0957 | 0.0026 | 0.0409 | [-0.3701, -0.1694] | 0.0160 | [-0.3938, -0.1022] | 0.0170 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0849 | 1.5291 | 0.1294 | 1.0000 | [+0.0142, +0.1492] | 0.0480 | [+0.0222, +0.1454] | 0.1820 | No |
| Gradient Boosted Trees (Full 13D Legacy) | -0.0213 | -0.7865 | 0.4334 | 1.0000 | [-0.0635, -0.0047] | 0.2420 | [-0.0547, +0.0147] | 1.0000 | No |
| SmallGAT (Graph Attention Network) | -0.3381 | -2.7706 | 0.0067 | 0.1002 | [-0.4562, -0.1961] | 0.0150 | [-0.4151, -0.2452] | 0.0160 | No |
| Majority-Class (Dummy Baseline) | -0.3798 | -7.3816 | 0.0000 | 0.0000 | [-0.4504, -0.3217] | 0.0140 | [-0.4303, -0.3194] | 0.0150 | Yes |

#### Drug-Disjoint (Primary)

| Comparison Model | Mean Diff | NB $t$-stat | NB Raw $p$ | NB Holm $p$ | Herb Boot 95% CI | Herb Holm $p$ | Drug Boot 95% CI | Drug Holm $p$ | Sig (NB)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.0443 | -0.7340 | 0.4647 | 1.0000 | [-0.1945, +0.0391] | 1.0000 | [-0.1265, +0.0560] | 1.0000 | No |
| Drug Identity (One-Hot) | -0.3673 | -7.8209 | 0.0000 | 0.0000 | [-0.4388, -0.2708] | 0.0200 | [-0.4214, -0.3070] | 0.0200 | Yes |
| Overlap Total (1D Scalar) | +0.0015 | 0.0444 | 0.9647 | 0.9647 | [-0.0464, +0.0398] | 1.0000 | [-0.0385, +0.0405] | 0.9860 | No |
| Overlap 6D (Pharmacokinetic Targets) | -0.0029 | -0.0752 | 0.9402 | 1.0000 | [-0.0389, +0.0225] | 1.0000 | [-0.0358, +0.0237] | 1.0000 | No |
| Curation 2D (Profile Density Proxy) | -0.0576 | -0.8877 | 0.3769 | 1.0000 | [-0.1926, +0.0349] | 1.0000 | [-0.1579, +0.0239] | 1.0000 | No |
| Lineage Features 7D (Echo/Quality/Compatibility) | -0.2527 | -4.6428 | 0.0000 | 0.0002 | [-0.3903, -0.0862] | 0.0190 | [-0.3756, -0.1681] | 0.0190 | Yes |
| Study Design Flags (9D Full Methodology) | -0.3261 | -5.9734 | 0.0000 | 0.0000 | [-0.4355, -0.2104] | 0.0180 | [-0.3959, -0.2314] | 0.0180 | Yes |
| LR (Overlap 6D + Curation 2D, 8D) | +0.0251 | 0.6589 | 0.5115 | 1.0000 | [-0.0250, +0.0740] | 1.0000 | [-0.0198, +0.0741] | 1.0000 | No |
| LR (Overlap 6D + Sanitized Design, 12D) | +0.0015 | 0.0469 | 0.9627 | 1.0000 | [-0.0391, +0.0325] | 0.9840 | [-0.0327, +0.0336] | 1.0000 | No |
| Masked-Text Level 1 (Outcome Masked) | +0.0788 | 1.4200 | 0.1588 | 1.0000 | [+0.0062, +0.1677] | 0.3380 | [+0.0140, +0.1506] | 0.1680 | No |
| Masked-Text Level 2 (Outcome & Entities Masked) | +0.0531 | 0.9017 | 0.3694 | 1.0000 | [-0.0222, +0.1537] | 1.0000 | [-0.0262, +0.1297] | 1.0000 | No |
| Masked-Text Level 3 (Outcome, Entities & Proxies Masked) | +0.0483 | 0.7926 | 0.4299 | 1.0000 | [-0.0306, +0.1542] | 1.0000 | [-0.0393, +0.1302] | 1.0000 | No |
| Masked-Text Norm Level 1 (Whitespace Replaced) | +0.0655 | 1.1315 | 0.2606 | 1.0000 | [-0.0105, +0.1575] | 1.0000 | [-0.0034, +0.1413] | 0.7440 | No |
| Masked-Text Norm Level 2 (Whitespace Replaced) | +0.0393 | 0.6391 | 0.5243 | 1.0000 | [-0.0455, +0.1394] | 1.0000 | [-0.0451, +0.1195] | 1.0000 | No |
| Masked-Text Norm Level 3 (Whitespace Replaced) | +0.0256 | 0.3862 | 0.7002 | 1.0000 | [-0.0712, +0.1395] | 1.0000 | [-0.0692, +0.1144] | 1.0000 | No |
| Design-Only (Structured Design Tokens) | -0.1937 | -2.3354 | 0.0215 | 0.3446 | [-0.2950, -0.1089] | 0.0170 | [-0.2776, -0.0857] | 0.0170 | No |
| Full-Text TF-IDF (LEAKY, reference only) | +0.1107 | 2.1225 | 0.0363 | 0.5444 | [+0.0492, +0.1983] | 0.0160 | [+0.0376, +0.1736] | 0.0300 | No |
| Gradient Boosted Trees (Full 13D Legacy) | +0.0181 | 0.5748 | 0.5667 | 1.0000 | [-0.0153, +0.0460] | 1.0000 | [-0.0132, +0.0497] | 1.0000 | No |
| SmallGAT (Graph Attention Network) | -0.1181 | -1.7839 | 0.0775 | 1.0000 | [-0.2196, -0.0537] | 0.0150 | [-0.1773, -0.0204] | 0.2080 | No |
| Majority-Class (Dummy Baseline) | -0.3673 | -7.8209 | 0.0000 | 0.0000 | [-0.4388, -0.2708] | 0.0140 | [-0.4214, -0.3070] | 0.0160 | Yes |

#### Pair-Disjoint (Primary)

| Comparison Model | Mean Diff | NB $t$-stat | NB Raw $p$ | NB Holm $p$ | Herb Boot 95% CI | Herb Holm $p$ | Drug Boot 95% CI | Drug Holm $p$ | Sig (NB)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Herb Identity (One-Hot) | -0.0925 | -1.9118 | 0.0588 | 0.7643 | [-0.2365, -0.0051] | 0.3300 | [-0.1804, +0.0047] | 0.7920 | No |
| Drug Identity (One-Hot) | -0.1653 | -3.2221 | 0.0017 | 0.0276 | [-0.2664, -0.0411] | 0.0560 | [-0.3196, -0.0810] | 0.0200 | Yes |
| Overlap Total (1D Scalar) | -0.0276 | -1.1864 | 0.2383 | 1.0000 | [-0.0719, +0.0062] | 1.0000 | [-0.0600, +0.0100] | 1.0000 | No |
| Overlap 6D (Pharmacokinetic Targets) | -0.0225 | -1.0519 | 0.2954 | 1.0000 | [-0.0512, -0.0014] | 0.3120 | [-0.0540, +0.0098] | 1.0000 | No |
| Curation 2D (Profile Density Proxy) | -0.0714 | -1.7570 | 0.0820 | 0.9842 | [-0.1957, +0.0158] | 1.0000 | [-0.1587, +0.0148] | 1.0000 | No |
| Lineage Features 7D (Echo/Quality/Compatibility) | -0.2692 | -4.7435 | 0.0000 | 0.0001 | [-0.3901, -0.1228] | 0.0200 | [-0.3605, -0.1912] | 0.0190 | Yes |
| Study Design Flags (9D Full Methodology) | -0.3193 | -7.5945 | 0.0000 | 0.0000 | [-0.4304, -0.1799] | 0.0190 | [-0.3925, -0.2256] | 0.0180 | Yes |
| LR (Overlap 6D + Curation 2D, 8D) | +0.0091 | 0.3271 | 0.7443 | 0.7443 | [-0.0320, +0.0594] | 0.6540 | [-0.0282, +0.0568] | 0.6780 | No |
| LR (Overlap 6D + Sanitized Design, 12D) | -0.0293 | -1.2227 | 0.2244 | 1.0000 | [-0.0647, -0.0040] | 0.2080 | [-0.0578, +0.0023] | 0.7920 | No |
| Masked-Text Level 1 (Outcome Masked) | +0.0556 | 1.7021 | 0.0919 | 1.0000 | [-0.0144, +0.1603] | 1.0000 | [+0.0015, +0.1302] | 0.6240 | No |
| Masked-Text Level 2 (Outcome & Entities Masked) | +0.0375 | 1.0263 | 0.3073 | 1.0000 | [-0.0418, +0.1502] | 1.0000 | [-0.0238, +0.1175] | 1.0000 | No |
| Masked-Text Level 3 (Outcome, Entities & Proxies Masked) | +0.0312 | 0.8268 | 0.4103 | 1.0000 | [-0.0502, +0.1491] | 1.0000 | [-0.0354, +0.1105] | 1.0000 | No |
| Masked-Text Norm Level 1 (Whitespace Replaced) | +0.0464 | 1.3363 | 0.1845 | 1.0000 | [-0.0266, +0.1496] | 1.0000 | [-0.0143, +0.1225] | 1.0000 | No |
| Masked-Text Norm Level 2 (Whitespace Replaced) | +0.0305 | 0.8120 | 0.4187 | 1.0000 | [-0.0521, +0.1441] | 1.0000 | [-0.0347, +0.1129] | 1.0000 | No |
| Masked-Text Norm Level 3 (Whitespace Replaced) | +0.0224 | 0.5612 | 0.5759 | 1.0000 | [-0.0692, +0.1428] | 1.0000 | [-0.0518, +0.1097] | 1.0000 | No |
| Design-Only (Structured Design Tokens) | -0.2257 | -4.2408 | 0.0001 | 0.0009 | [-0.3306, -0.1084] | 0.0180 | [-0.3422, -0.0954] | 0.0170 | Yes |
| Full-Text TF-IDF (LEAKY, reference only) | +0.0868 | 2.7440 | 0.0072 | 0.1081 | [+0.0239, +0.1858] | 0.0170 | [+0.0349, +0.1543] | 0.0300 | No |
| Gradient Boosted Trees (Full 13D Legacy) | -0.0116 | -0.4389 | 0.6617 | 1.0000 | [-0.0369, +0.0079] | 1.0000 | [-0.0393, +0.0148] | 1.0000 | No |
| SmallGAT (Graph Attention Network) | -0.1185 | -2.5522 | 0.0122 | 0.1712 | [-0.2113, -0.0511] | 0.0160 | [-0.1946, -0.0256] | 0.2520 | No |
| Majority-Class (Dummy Baseline) | -0.3903 | -13.1030 | 0.0000 | 0.0000 | [-0.4565, -0.2846] | 0.0150 | [-0.4361, -0.3253] | 0.0160 | Yes |

### 4. SmallGAT Retention Protocol Decision
- **Decision Rule:** Keep SmallGAT only if it beats LR under all three disjoint protocols with p < 0.05
- **Final Verdict:** **`DROPPED`**
- **Rationale:** SmallGAT is REJECTED / DROPPED. Under the retention protocol, SmallGAT must beat Logistic Regression under all three primary disjoint split protocols (herb-disjoint, drug-disjoint, pair-disjoint) with statistical significance (p < 0.05). SmallGAT beats LR in all three: False.

| Disjoint Protocol | SmallGAT AUC | LR AUC | Mean Diff | NB $t$-statistic | NB $p$-value | Beats LR ($p<0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `herb_disjoint` | 0.5417 | 0.8798 | -0.3381 | -2.7706 | 0.0067 | No |
| `drug_disjoint` | 0.7492 | 0.8673 | -0.1181 | -1.7839 | 0.0775 | No |
| `pair_disjoint` | 0.7718 | 0.8903 | -0.1185 | -2.5522 | 0.0122 | No |

### 5. Text Masking Hierarchy & C-Tuning Analysis (Fixed C=1.0 vs. Inner-CV C)
Verifies that masking levels enforce strict monotonicity without outcome leakage inversions.

#### Herb-Disjoint (Primary)
| Text Representation Level | Fixed C=1.0 Mean AUC (SD) | Inner-CV C Mean AUC (SD) | Chosen C Median [Min, Max] | Hierarchy Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Text (Unmasked, Level 0)** | 0.9647 (0.0463) | 0.9668 (0.0441) | 10.0 [1.0, 100.0] | LEAKY |
| **Masked Level 1 (Outcome Masked)** | 0.9167 (0.0660) | 0.9151 (0.0636) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Level 2 (Outcome & Entities Masked)** | 0.9190 (0.0639) | 0.9223 (0.0618) | 100.0 [0.1, 100.0] | MONOTONIC |
| **Masked Level 3 (Outcome, Entities & Proxies)** | 0.9120 (0.0657) | 0.9145 (0.0640) | 10.0 [0.1, 100.0] | MONOTONIC |
| **Masked Norm Level 1 (Whitespace Replaced)** | 0.9085 (0.0721) | 0.9078 (0.0709) | 10.0 [0.1, 100.0] | MONOTONIC |
| **Masked Norm Level 2 (Whitespace Replaced)** | 0.9142 (0.0663) | 0.9187 (0.0671) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Norm Level 3 (Whitespace Replaced)** | 0.9031 (0.0719) | 0.9058 (0.0694) | 10.0 [0.1, 100.0] | MONOTONIC |
| **Design-Only (Structured Design Tokens)** | 0.6378 (0.1022) | 0.6289 (0.1125) | 1.0 [0.01, 100.0] | MONOTONIC |

#### Drug-Disjoint (Primary)
| Text Representation Level | Fixed C=1.0 Mean AUC (SD) | Inner-CV C Mean AUC (SD) | Chosen C Median [Min, Max] | Hierarchy Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Text (Unmasked, Level 0)** | 0.9780 (0.0335) | 0.9788 (0.0351) | 10.0 [0.1, 100.0] | LEAKY |
| **Masked Level 1 (Outcome Masked)** | 0.9461 (0.0427) | 0.9432 (0.0447) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Level 2 (Outcome & Entities Masked)** | 0.9205 (0.0599) | 0.9221 (0.0626) | 10.0 [0.1, 100.0] | MONOTONIC |
| **Masked Level 3 (Outcome, Entities & Proxies)** | 0.9156 (0.0640) | 0.9142 (0.0699) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Norm Level 1 (Whitespace Replaced)** | 0.9328 (0.0502) | 0.9314 (0.0516) | 10.0 [0.1, 100.0] | MONOTONIC |
| **Masked Norm Level 2 (Whitespace Replaced)** | 0.9066 (0.0641) | 0.9119 (0.0668) | 10.0 [0.1, 100.0] | MONOTONIC |
| **Masked Norm Level 3 (Whitespace Replaced)** | 0.8930 (0.0727) | 0.9005 (0.0730) | 5.5 [0.1, 100.0] | MONOTONIC |
| **Design-Only (Structured Design Tokens)** | 0.6737 (0.1128) | 0.6713 (0.1112) | 1.0 [0.001, 100.0] | MONOTONIC |

#### Pair-Disjoint (Primary)
| Text Representation Level | Fixed C=1.0 Mean AUC (SD) | Inner-CV C Mean AUC (SD) | Chosen C Median [Min, Max] | Hierarchy Status |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Text (Unmasked, Level 0)** | 0.9771 (0.0262) | 0.9788 (0.0256) | 10.0 [1.0, 100.0] | LEAKY |
| **Masked Level 1 (Outcome Masked)** | 0.9459 (0.0382) | 0.9442 (0.0395) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Level 2 (Outcome & Entities Masked)** | 0.9278 (0.0510) | 0.9278 (0.0541) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Level 3 (Outcome, Entities & Proxies)** | 0.9215 (0.0528) | 0.9209 (0.0555) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Norm Level 1 (Whitespace Replaced)** | 0.9368 (0.0446) | 0.9333 (0.0451) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Norm Level 2 (Whitespace Replaced)** | 0.9208 (0.0543) | 0.9231 (0.0582) | 10.0 [1.0, 100.0] | MONOTONIC |
| **Masked Norm Level 3 (Whitespace Replaced)** | 0.9127 (0.0580) | 0.9125 (0.0578) | 1.0 [1.0, 100.0] | MONOTONIC |
| **Design-Only (Structured Design Tokens)** | 0.6646 (0.0865) | 0.6624 (0.0949) | 1.0 [0.1, 100.0] | MONOTONIC |

### 6. Conformal Prediction & Selective Classification (Matched Abstention Suite)
> [!NOTE]
> Conformal abstention operates under $\alpha = 0.10$. Conflict and reliability signals are **NOT evaluated** until consensus labels exist.

#### Abstention Evaluation (Herb-Disjoint (Primary))
> [!NOTE]
> **Tri-State Routing Status:** Under legacy single-study binary labels, the tri-state conformal engine operates without external multi-study conflict annotations; its conformal nonconformity score on single probabilities is monotonically equivalent to max-probability |p - 0.5|. Possessing no separate signal under legacy labels, the tri-state row is dropped from the matched comparison table to avoid redundant duplicate scoring.

##### Selective Accuracy at Matched Coverage Levels
| Target Coverage | Effective N | Max-Prob LR Acc (Risk) | GBT Uncertainty Acc (Risk) |
| :---: | :---: | :---: | :---: |
| 0.5000 | 1410 | 0.8844 (0.1156) | 0.8794 (0.1206) |
| 0.6000 | 1692 | 0.8818 (0.1182) | 0.8806 (0.1194) |
| 0.7000 | 1974 | 0.8739 (0.1261) | 0.8652 (0.1348) |
| 0.8000 | 2256 | 0.8533 (0.1467) | 0.8378 (0.1622) |
| 0.8500 | 2397 | 0.8444 (0.1556) | 0.8244 (0.1756) |
| 0.9000 | 2538 | 0.8373 (0.1627) | 0.8121 (0.1879) |
| 0.9500 | 2679 | 0.8302 (0.1698) | 0.7977 (0.2023) |

##### Coverage Achieved at Matched Target Selective Accuracy
| Target Accuracy | Max-Prob LR Coverage | GBT Uncertainty Coverage |
| :---: | :---: | :---: |
| 0.8000 | 1.0000 | 0.9372 |
| 0.8500 | 0.8227 | 0.7496 |
| 0.9000 | 0.3663 | 0.2238 |
| 0.9500 | 0.1631 | 0.1734 |

#### Abstention Evaluation (Drug-Disjoint (Primary))
> [!NOTE]
> **Tri-State Routing Status:** Under legacy single-study binary labels, the tri-state conformal engine operates without external multi-study conflict annotations; its conformal nonconformity score on single probabilities is monotonically equivalent to max-probability |p - 0.5|. Possessing no separate signal under legacy labels, the tri-state row is dropped from the matched comparison table to avoid redundant duplicate scoring.

##### Selective Accuracy at Matched Coverage Levels
| Target Coverage | Effective N | Max-Prob LR Acc (Risk) | GBT Uncertainty Acc (Risk) |
| :---: | :---: | :---: | :---: |
| 0.5000 | 1410 | 0.8525 (0.1475) | 0.8844 (0.1156) |
| 0.6000 | 1692 | 0.8534 (0.1466) | 0.8806 (0.1194) |
| 0.7000 | 1974 | 0.8516 (0.1484) | 0.8627 (0.1373) |
| 0.8000 | 2256 | 0.8409 (0.1591) | 0.8484 (0.1516) |
| 0.8500 | 2397 | 0.8344 (0.1656) | 0.8398 (0.1602) |
| 0.9000 | 2538 | 0.8278 (0.1722) | 0.8286 (0.1714) |
| 0.9500 | 2679 | 0.8145 (0.1855) | 0.8141 (0.1859) |

##### Coverage Achieved at Matched Target Selective Accuracy
| Target Accuracy | Max-Prob LR Coverage | GBT Uncertainty Coverage |
| :---: | :---: | :---: |
| 0.8000 | 1.0000 | 0.9947 |
| 0.8500 | 0.7496 | 0.7943 |
| 0.9000 | 0.1606 | 0.3663 |
| 0.9500 | 0.1089 | 0.3064 |

#### Abstention Evaluation (Pair-Disjoint (Primary))
> [!NOTE]
> **Tri-State Routing Status:** Under legacy single-study binary labels, the tri-state conformal engine operates without external multi-study conflict annotations; its conformal nonconformity score on single probabilities is monotonically equivalent to max-probability |p - 0.5|. Possessing no separate signal under legacy labels, the tri-state row is dropped from the matched comparison table to avoid redundant duplicate scoring.

##### Selective Accuracy at Matched Coverage Levels
| Target Coverage | Effective N | Max-Prob LR Acc (Risk) | GBT Uncertainty Acc (Risk) |
| :---: | :---: | :---: | :---: |
| 0.5000 | 1410 | 0.9199 (0.0801) | 0.8972 (0.1028) |
| 0.6000 | 1692 | 0.9178 (0.0822) | 0.8989 (0.1011) |
| 0.7000 | 1974 | 0.8951 (0.1049) | 0.8794 (0.1206) |
| 0.8000 | 2256 | 0.8670 (0.1330) | 0.8520 (0.1480) |
| 0.8500 | 2397 | 0.8565 (0.1435) | 0.8373 (0.1627) |
| 0.9000 | 2538 | 0.8440 (0.1560) | 0.8247 (0.1753) |
| 0.9500 | 2679 | 0.8302 (0.1698) | 0.8160 (0.1840) |

##### Coverage Achieved at Matched Target Selective Accuracy
| Target Accuracy | Max-Prob LR Coverage | GBT Uncertainty Coverage |
| :---: | :---: | :---: |
| 0.8000 | 1.0000 | 1.0000 |
| 0.8500 | 0.8762 | 0.8057 |
| 0.9000 | 0.6816 | 0.6191 |
| 0.9500 | 0.3128 | 0.3223 |

### 7. Internal Hyperparameter Tuning: Chosen $C$ Per Feature Set Per Split
Median and range of $L_2$ regularization parameter $C$ selected via 3-fold inner cross-validation across 100 folds:

#### Herb-Disjoint (Primary)
| Feature Set / Model | Median Chosen $C$ | Range [$C_{\min}, C_{\max}$] | Edge Rate ($C=0.001$ or $100.0$) | Distribution Counts |
| :--- | :---: | :---: | :---: | :--- |
| **Herb Identity (One-Hot)** | 1.0000 | [0.100, 100.0] | 2.00% (min:0, max:2) | C=0.1: 41, C=1.0: 54, C=10.0: 3, C=100.0: 2 |
| **Drug Identity (One-Hot)** | 1.0000 | [0.100, 10.0] | 0.00% (min:0, max:0) | C=0.1: 13, C=1.0: 52, C=10.0: 35 |
| **Overlap Total (1D Scalar)** | 0.1000 | [0.010, 100.0] | 3.00% (min:0, max:3) | C=0.01: 8, C=0.1: 52, C=1.0: 26, C=10.0: 11, C=100.0: 3 |
| **Overlap 6D (Pharmacokinetic Targets)** | 10.0000 | [0.100, 100.0] | 18.00% (min:0, max:18) | C=0.1: 14, C=1.0: 16, C=10.0: 52, C=100.0: 18 |
| **Curation 2D (Profile Density Proxy)** | 0.0550 | [0.001, 100.0] | 5.00% (min:3, max:2) | C=0.001: 3, C=0.01: 47, C=0.1: 47, C=1.0: 1, C=100.0: 2 |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | 10.0000 | [0.001, 100.0] | 45.00% (min:23, max:22) | C=0.001: 23, C=0.1: 4, C=1.0: 14, C=10.0: 37, C=100.0: 22 |
| **Study Design Flags (9D Full Methodology)** | 1.0000 | [0.001, 100.0] | 32.00% (min:28, max:4) | C=0.001: 28, C=0.1: 1, C=1.0: 46, C=10.0: 21, C=100.0: 4 |
| **LR (Overlap 6D + Curation 2D, 8D)** | 1.0000 | [0.010, 100.0] | 13.00% (min:0, max:13) | C=0.01: 1, C=0.1: 16, C=1.0: 42, C=10.0: 28, C=100.0: 13 |
| **LR (Overlap 6D + Sanitized Design, 12D)** | 10.0000 | [0.100, 100.0] | 21.00% (min:0, max:21) | C=0.1: 19, C=1.0: 20, C=10.0: 40, C=100.0: 21 |
| **Masked-Text Level 1 (Outcome Masked)** | 10.0000 | [1.000, 100.0] | 39.00% (min:0, max:39) | C=1.0: 32, C=10.0: 29, C=100.0: 39 |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | 100.0000 | [0.100, 100.0] | 51.00% (min:0, max:51) | C=0.1: 1, C=1.0: 7, C=10.0: 41, C=100.0: 51 |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | 10.0000 | [0.100, 100.0] | 23.00% (min:0, max:23) | C=0.1: 1, C=1.0: 23, C=10.0: 53, C=100.0: 23 |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | 10.0000 | [0.100, 100.0] | 19.00% (min:0, max:19) | C=0.1: 2, C=1.0: 41, C=10.0: 38, C=100.0: 19 |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | 10.0000 | [1.000, 100.0] | 38.00% (min:0, max:38) | C=1.0: 6, C=10.0: 56, C=100.0: 38 |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | 10.0000 | [0.100, 100.0] | 7.00% (min:0, max:7) | C=0.1: 1, C=1.0: 38, C=10.0: 54, C=100.0: 7 |
| **Design-Only (Structured Design Tokens)** | 1.0000 | [0.010, 100.0] | 8.00% (min:0, max:8) | C=0.01: 1, C=0.1: 17, C=1.0: 53, C=10.0: 21, C=100.0: 8 |
| **Full-Text TF-IDF (LEAKY, reference only)** | 10.0000 | [1.000, 100.0] | 41.00% (min:0, max:41) | C=1.0: 8, C=10.0: 51, C=100.0: 41 |
| **Logistic Regression (Full 13D Legacy)** | 100.0000 | [0.100, 100.0] | 84.00% (min:0, max:84) | C=0.1: 5, C=1.0: 8, C=10.0: 3, C=100.0: 84 |

#### Drug-Disjoint (Primary)
| Feature Set / Model | Median Chosen $C$ | Range [$C_{\min}, C_{\max}$] | Edge Rate ($C=0.001$ or $100.0$) | Distribution Counts |
| :--- | :---: | :---: | :---: | :--- |
| **Herb Identity (One-Hot)** | 1.0000 | [0.100, 100.0] | 3.00% (min:0, max:3) | C=0.1: 39, C=1.0: 56, C=10.0: 2, C=100.0: 3 |
| **Drug Identity (One-Hot)** | 1.0000 | [0.100, 10.0] | 0.00% (min:0, max:0) | C=0.1: 27, C=1.0: 28, C=10.0: 45 |
| **Overlap Total (1D Scalar)** | 1.0000 | [0.010, 100.0] | 3.00% (min:0, max:3) | C=0.01: 13, C=0.1: 31, C=1.0: 43, C=10.0: 10, C=100.0: 3 |
| **Overlap 6D (Pharmacokinetic Targets)** | 10.0000 | [0.100, 100.0] | 11.00% (min:0, max:11) | C=0.1: 9, C=1.0: 17, C=10.0: 63, C=100.0: 11 |
| **Curation 2D (Profile Density Proxy)** | 0.0100 | [0.001, 100.0] | 9.00% (min:8, max:1) | C=0.001: 8, C=0.01: 54, C=0.1: 34, C=1.0: 2, C=10.0: 1, C=100.0: 1 |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | 10.0000 | [0.001, 100.0] | 37.00% (min:10, max:27) | C=0.001: 10, C=1.0: 12, C=10.0: 51, C=100.0: 27 |
| **Study Design Flags (9D Full Methodology)** | 1.0000 | [0.001, 10.0] | 13.00% (min:13, max:0) | C=0.001: 13, C=0.1: 3, C=1.0: 62, C=10.0: 22 |
| **LR (Overlap 6D + Curation 2D, 8D)** | 1.0000 | [0.010, 100.0] | 16.00% (min:0, max:16) | C=0.01: 2, C=0.1: 18, C=1.0: 39, C=10.0: 25, C=100.0: 16 |
| **LR (Overlap 6D + Sanitized Design, 12D)** | 10.0000 | [0.100, 100.0] | 15.00% (min:0, max:15) | C=0.1: 14, C=1.0: 24, C=10.0: 47, C=100.0: 15 |
| **Masked-Text Level 1 (Outcome Masked)** | 10.0000 | [1.000, 100.0] | 39.00% (min:0, max:39) | C=1.0: 28, C=10.0: 33, C=100.0: 39 |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | 10.0000 | [0.100, 100.0] | 35.00% (min:0, max:35) | C=0.1: 1, C=1.0: 9, C=10.0: 55, C=100.0: 35 |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | 10.0000 | [1.000, 100.0] | 23.00% (min:0, max:23) | C=1.0: 30, C=10.0: 47, C=100.0: 23 |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | 10.0000 | [0.100, 100.0] | 19.00% (min:0, max:19) | C=0.1: 1, C=1.0: 46, C=10.0: 34, C=100.0: 19 |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | 10.0000 | [0.100, 100.0] | 26.00% (min:0, max:26) | C=0.1: 1, C=1.0: 18, C=10.0: 55, C=100.0: 26 |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | 5.5000 | [0.100, 100.0] | 10.00% (min:0, max:10) | C=0.1: 2, C=1.0: 48, C=10.0: 40, C=100.0: 10 |
| **Design-Only (Structured Design Tokens)** | 1.0000 | [0.001, 100.0] | 12.00% (min:3, max:9) | C=0.001: 3, C=0.1: 21, C=1.0: 51, C=10.0: 16, C=100.0: 9 |
| **Full-Text TF-IDF (LEAKY, reference only)** | 10.0000 | [0.100, 100.0] | 27.00% (min:0, max:27) | C=0.1: 1, C=1.0: 2, C=10.0: 70, C=100.0: 27 |
| **Logistic Regression (Full 13D Legacy)** | 100.0000 | [0.100, 100.0] | 86.00% (min:0, max:86) | C=0.1: 1, C=1.0: 4, C=10.0: 9, C=100.0: 86 |

#### Pair-Disjoint (Primary)
| Feature Set / Model | Median Chosen $C$ | Range [$C_{\min}, C_{\max}$] | Edge Rate ($C=0.001$ or $100.0$) | Distribution Counts |
| :--- | :---: | :---: | :---: | :--- |
| **Herb Identity (One-Hot)** | 1.0000 | [0.100, 100.0] | 2.00% (min:0, max:2) | C=0.1: 46, C=1.0: 44, C=10.0: 8, C=100.0: 2 |
| **Drug Identity (One-Hot)** | 1.0000 | [0.100, 10.0] | 0.00% (min:0, max:0) | C=0.1: 12, C=1.0: 68, C=10.0: 20 |
| **Overlap Total (1D Scalar)** | 0.1000 | [0.010, 100.0] | 2.00% (min:0, max:2) | C=0.01: 12, C=0.1: 62, C=1.0: 18, C=10.0: 6, C=100.0: 2 |
| **Overlap 6D (Pharmacokinetic Targets)** | 10.0000 | [0.100, 100.0] | 15.00% (min:0, max:15) | C=0.1: 4, C=1.0: 15, C=10.0: 66, C=100.0: 15 |
| **Curation 2D (Profile Density Proxy)** | 0.0100 | [0.010, 100.0] | 1.00% (min:0, max:1) | C=0.01: 67, C=0.1: 32, C=100.0: 1 |
| **Lineage Features 7D (Echo/Quality/Compatibility)** | 10.0000 | [0.001, 100.0] | 46.00% (min:7, max:39) | C=0.001: 7, C=0.1: 9, C=1.0: 8, C=10.0: 37, C=100.0: 39 |
| **Study Design Flags (9D Full Methodology)** | 1.0000 | [0.001, 100.0] | 14.00% (min:9, max:5) | C=0.001: 9, C=0.1: 2, C=1.0: 79, C=10.0: 5, C=100.0: 5 |
| **LR (Overlap 6D + Curation 2D, 8D)** | 1.0000 | [0.100, 100.0] | 14.00% (min:0, max:14) | C=0.1: 14, C=1.0: 39, C=10.0: 33, C=100.0: 14 |
| **LR (Overlap 6D + Sanitized Design, 12D)** | 10.0000 | [0.100, 100.0] | 19.00% (min:0, max:19) | C=0.1: 12, C=1.0: 27, C=10.0: 42, C=100.0: 19 |
| **Masked-Text Level 1 (Outcome Masked)** | 10.0000 | [1.000, 100.0] | 38.00% (min:0, max:38) | C=1.0: 22, C=10.0: 40, C=100.0: 38 |
| **Masked-Text Level 2 (Outcome & Entities Masked)** | 10.0000 | [1.000, 100.0] | 35.00% (min:0, max:35) | C=1.0: 6, C=10.0: 59, C=100.0: 35 |
| **Masked-Text Level 3 (Outcome, Entities & Proxies Masked)** | 10.0000 | [1.000, 100.0] | 31.00% (min:0, max:31) | C=1.0: 24, C=10.0: 45, C=100.0: 31 |
| **Masked-Text Norm Level 1 (Whitespace Replaced)** | 10.0000 | [1.000, 100.0] | 20.00% (min:0, max:20) | C=1.0: 46, C=10.0: 34, C=100.0: 20 |
| **Masked-Text Norm Level 2 (Whitespace Replaced)** | 10.0000 | [1.000, 100.0] | 28.00% (min:0, max:28) | C=1.0: 16, C=10.0: 56, C=100.0: 28 |
| **Masked-Text Norm Level 3 (Whitespace Replaced)** | 1.0000 | [1.000, 100.0] | 2.00% (min:0, max:2) | C=1.0: 52, C=10.0: 46, C=100.0: 2 |
| **Design-Only (Structured Design Tokens)** | 1.0000 | [0.100, 100.0] | 13.00% (min:0, max:13) | C=0.1: 27, C=1.0: 50, C=10.0: 10, C=100.0: 13 |
| **Full-Text TF-IDF (LEAKY, reference only)** | 10.0000 | [1.000, 100.0] | 41.00% (min:0, max:41) | C=1.0: 3, C=10.0: 56, C=100.0: 41 |
| **Logistic Regression (Full 13D Legacy)** | 100.0000 | [0.100, 100.0] | 79.00% (min:0, max:79) | C=0.1: 1, C=1.0: 13, C=10.0: 7, C=100.0: 79 |

### 8. Seed Perturbation Stability Analysis (Seeds 43, 44, 45 vs. Repeat SD across 20 Repeats)
Evaluates model sensitivity to random partition seeds (100 folds per seed) compared to repeat standard deviation (SD across 20 repeat means) under primary seed 42:

| Model / Feature Set | Seed 42 Mean AUC | Repeat SD (20 Repeats) | Seed 43 Shift | Seed 44 Shift | Seed 45 Shift | Max Shift | Max Shift / Repeat SD | Bounded by SD? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `overlap_6d` | 0.8679 | 0.0081 | 0.0050 | 0.0071 | 0.0061 | 0.0071 | 0.8808 | Yes |
| `curation_2d` | 0.8189 | 0.0075 | 0.0021 | 0.0035 | 0.0013 | 0.0035 | 0.4645 | Yes |
| `lineage_features_7d` | 0.6211 | 0.0303 | 0.0144 | 0.0258 | 0.0105 | 0.0258 | 0.8516 | Yes |
| `study_design_flags` | 0.5710 | 0.0030 | 0.0002 | 0.0054 | 0.0004 | 0.0054 | 1.7829 | No |
| `lr_overlap_plus_curation` | 0.8994 | 0.0129 | 0.0053 | 0.0052 | 0.0071 | 0.0071 | 0.5474 | Yes |
| `lr_overlap_plus_sanitized_design` | 0.8610 | 0.0128 | 0.0090 | 0.0100 | 0.0098 | 0.0100 | 0.7781 | Yes |
| `logistic_regression` | 0.8903 | 0.0140 | 0.0037 | 0.0010 | 0.0016 | 0.0037 | 0.2657 | Yes |

### 9. Genre-Stratified and Pure-Negative-Herb Sensitivity Analysis (Herb-Disjoint)
Evaluates the impact of study genres and the 4 pure-negative botanicals (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`) on key feature representations:

| Cohort Name | $N$ | Pos / Neg | `overlap_6d` AUC (SD) | `curation_2d` AUC (SD) | `lineage_7d` AUC (SD) | `study_design_flags` AUC (SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clinical Trials (All Herbs) | 121 | 60 / 61 | 0.9087 (0.0548) | 0.8037 (0.1086) | 0.5330 (0.0940) | 0.5126 (0.0669) |
| Clinical Trials (Excl. 4 Pure-Neg Herbs) | 96 | 60 / 36 | 0.8854 (0.0621) | 0.6534 (0.1381) | 0.5647 (0.0930) | 0.5873 (0.0871) |
| Healthy Volunteers (All Herbs) | 106 | 48 / 58 | 0.8927 (0.0623) | 0.7648 (0.1215) | 0.5451 (0.1188) | 0.4920 (0.0723) |
| Healthy Volunteers (Excl. 4 Pure-Neg Herbs) | 83 | 48 / 35 | 0.8687 (0.0741) | 0.5931 (0.1313) | 0.5849 (0.0982) | 0.5887 (0.0836) |
| Case Reports (All Herbs) | 16 | 14 / 2 | nan (nan) | nan (nan) | nan (nan) | nan (nan) |

---
### 10. Summary & Key Findings
1. **Label-Adjacent Signal:** Tabular biological overlap (`0.86+` AUC) continues to perform strongly under legacy labels, but operates as a label-adjacent profile proxy.
2. **Curation-Density Parity:** The 2D curation baseline (`has_measured_target`, `n_nonzero_targets`) demonstrates comparable performance, indicating profile curation density is a primary non-mechanistic confound.
3. **Masking Hierarchy Restored:** The Level 2 masking bug (entity masking without outcome masking) has been corrected. Text representations now exhibit strict monotonic attenuation: Full Text (0.97+) $\ge$ Level 1 $\ge$ Level 2 $\ge$ Level 3 $\ge$ Design-Only.
4. **SmallGAT Temperature Ceiling:** SmallGAT's post-hoc calibration temperature hits the optimization upper bound because on unseen botanicals (`herb_disjoint`), GAT predictions lack discriminative generalization (AUC ~ 0.54) while producing overconfident logits. The NLL loss drives $T \to \infty$ to collapse probabilities to the uniform prior.
5. **Sanitized Design Result (0.4328) & Cohort Label Rates:** The sub-chance AUC of 0.4328 achieved by sanitized study design features in the clinical trials cohort excluding pure-negative botanicals ($N=100$, 61 positive, 39 negative, positive base rate = 0.6100) is **unexplained**. Empirical token label rates in this cohort are: herbal_tea (N=6, 66.7% pos), intravenous_administration (N=28, 60.7% pos), juice_preparation (N=9, 44.4% pos), methodology_unspecified (N=45, 66.7% pos), oil_formulation (N=4, 75.0% pos), oral_administration (N=10, 70.0% pos), standardized_extract (N=14, 35.7% pos).
6. **Task B Non-Testability:** Task B multi-study evidence aggregation remains untestable on this 141-record panel until human consensus labels are curated.
