# PHASE 1 LEAKAGE PROBE REPORT
**Date:** 2026-10-27  
**Objective:** Empirically quantify whether label-adjacent polarity features (S_pos, S_neg, margin, N_raw_pos, N_raw_neg) leak ground-truth targets to models.

## Probe Results Table
| Model & Feature Set | ROC-AUC | PR-AUC | F1-Score | Accuracy | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `LogisticRegression (polarity_only (label-adjacent))` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.0090 |
| `GradientBoosting (polarity_only (label-adjacent))` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `LogisticRegression (quality_only (leakage-free))` | **0.5327** | 0.4854 | 0.6471 | 0.5385 | 0.2569 |
| `GradientBoosting (quality_only (leakage-free))` | **0.4911** | 0.4788 | 0.5000 | 0.5385 | 0.3000 |
| `LogisticRegression (full_12d)` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.0063 |
| `GradientBoosting (full_12d)` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Analysis & Decision
1. **Polarity-Only Performance:**
   - Logistic Regression on Polarity-Only: ROC-AUC = **1.0000**
   - Gradient Boosting on Polarity-Only: ROC-AUC = **1.0000**
   - **Conclusion:** A simple linear model using only the 5 polarity features achieves near-perfect classification on the held-out partition. This proves that feeding polarity features computed from test evaluation records constitutes direct target leakage.

2. **Quality-Only Performance (Leakage-Free):**
   - Logistic Regression on Quality-Only: ROC-AUC = **0.5327**
   - Gradient Boosting on Quality-Only: ROC-AUC = **0.4911**
   - **Conclusion:** Quality features alone (`echo_ratio`, `source_quality`, `context_compat`, `missingness`, `rho_reach`, `f_m_CYP`) evaluate evidentiary robustness and PK parameters without leaking the claim's polarity.

3. **Headline Model Policy:**
   - In accordance with Phase 1 instructions, **`quality_only` is made the default headline configuration** for `ReliabilityConditionedInteractionModel`.
   - The full 12D mode is retained and reported transparently alongside `quality_only` as an ablation.
