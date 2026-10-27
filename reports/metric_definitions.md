# Standardized Metric Definitions and Evaluation Protocol

This document provides mathematical definitions, exact denominators, operating thresholds, and statistical estimation methods for all metrics reported across the **Reliability-Conditioned Herb–Drug Interaction (HDI) Prediction Framework**.

---

## 1. Test Split Provenance and Denominators

All reported test metrics are evaluated on the canonical **study-disjoint test partition** derived from `data/benchmark/verified_botanical_hdi.csv` ($N = 141$ verified clinical interaction records, seed = 42, 60/20/20 partition ratio).

### Sample Counts and Class Balance
* **Total Benchmark Records ($N_{\text{total}}$):** 141
* **Training Partition ($N_{\text{train}}$):** 90 records across 90 primary studies
* **Validation Partition ($N_{\text{val}}$):** 25 records across 25 primary studies
* **Test Partition ($N_{\text{test}}$):** 26 records across 26 primary studies
* **Test Class Distribution:**
  * Positive (Interaction Present, $y = 1$): $N_{\text{pos}} = 12$
  * Negative (No Interaction Observed, $y = 0$): $N_{\text{neg}} = 14$
  * Conflicting ($y = 0.5$): 0 records in test partition (mapped to 0 if present)
* **Pair Comparisons for Ranking Metrics:** $N_{\text{pos}} \times N_{\text{neg}} = 12 \times 14 = 168$ distinct positive-negative pairs.

---

## 2. Metric Formulations and Denominators

### 2.1 Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
* **Definition:** Probability that a randomly drawn positive instance ($y=1$) is assigned a higher predicted interaction probability than a randomly drawn negative instance ($y=0$):
  $$\text{ROC-AUC} = \frac{1}{N_{\text{pos}} \cdot N_{\text{neg}}} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \left( \mathbb{I}[\hat{p}_i > \hat{p}_j] + 0.5 \cdot \mathbb{I}[\hat{p}_i = \hat{p}_j] \right)$$
* **Numerator:** Total rank points from pairwise comparisons where positive probability exceeds negative probability.
* **Denominator:** $N_{\text{pos}} \times N_{\text{neg}} = 12 \times 14 = 168$.
* **Headline Value (Reliability-Conditioned Quality-Only):** `0.8512` (exact: `0.8511904761904763`, 95% CI: `[0.6607, 0.9881]`).

### 2.2 Area Under the Precision-Recall Curve (PR-AUC / Average Precision)
* **Definition:** Weighted average of precisions across all recall operating thresholds:
  $$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) \cdot P_k$$
  where $P_k$ and $R_k$ are precision and recall at the $k$-th probability threshold.
* **Clinical Rationale:** Particularly sensitive to minority positive interaction detection in imbalanced botanical safety surveillance.
* **Denominator:** Evaluated trapezoidally across the unit recall continuum $[0, 1]$.
* **Headline Value:** `0.8794` (exact: `0.8793501375309157`, 95% CI: `[0.6723, 0.9875]`).

### 2.3 F1-Score (Binary Classification)
* **Definition:** Harmonic mean of precision and recall at decision threshold $\tau = 0.5$:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
* **Decision Rule:** $\hat{y}_i = \mathbb{I}[\hat{p}_i \ge 0.5]$.
* **Confusion Matrix at $\tau = 0.5$ ($N = 26$):**
  * True Positives ($\text{TP}$): 9
  * False Positives ($\text{FP}$): 2
  * False Negatives ($\text{FN}$): 3
  * True Negatives ($\text{TN}$): 12
* **Numerator:** $2 \cdot \text{TP} = 2 \times 9 = 18$.
* **Denominator:** $2 \cdot \text{TP} + \text{FP} + \text{FN} = 18 + 2 + 3 = 23$.
* **Headline Value:** $18 / 23 \approx 0.7826$ (exact: `0.782608695652174`, 95% CI: `[0.5556, 0.9231]`).

### 2.4 Accuracy
* **Definition:** Overall fraction of test samples correctly classified at threshold $\tau = 0.5$:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N_{\text{total}}}$$
* **Numerator:** $\text{TP} + \text{TN} = 9 + 12 = 21$.
* **Denominator:** $N_{\text{total}} = 26$.
* **Headline Value:** $21 / 26 \approx 0.8077$ (exact: `0.8076923076923077`, 95% CI: `[0.6538, 0.9231]`).

### 2.5 Precision (Positive Predictive Value)
* **Definition:** Fraction of declared interactions that are verified interactions:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* **Numerator:** $\text{TP} = 9$.
* **Denominator:** $\text{TP} + \text{FP} = 9 + 2 = 11$.
* **Headline Value:** $9 / 11 \approx 0.8182$ (exact: `0.8181818181818182`).

### 2.6 Recall (Sensitivity)
* **Definition:** Fraction of verified clinical interactions captured by the model:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **Numerator:** $\text{TP} = 9$.
* **Denominator:** $N_{\text{pos}} = 9 + 3 = 12$.
* **Headline Value:** $9 / 12 = 0.7500$ (exact: `0.75`).

### 2.7 Brier Score (Mean Squared Probability Error)
* **Definition:** Mean squared difference between predicted calibrated probability $\hat{p}_i$ and binary outcome $y_i \in \{0, 1\}$:
  $$\text{Brier Score} = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} (\hat{p}_i - y_i)^2$$
* **Numerator:** $\sum_{i=1}^{26} (\hat{p}_i - y_i)^2 \approx 4.0401$.
* **Denominator:** $N_{\text{total}} = 26$.
* **Headline Value:** `0.1554` (exact: `0.1553872972726822`, 95% CI: `[0.0886, 0.2289]`).

### 2.8 Class-Stratified Brier Scores ($BS_{\text{pos}}$, $BS_{\text{neg}}$)
* **Positive Class Brier ($BS_{\text{pos}}$):**
  $$BS_{\text{pos}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \mathcal{P}} (\hat{p}_i - 1)^2$$
  * Denominator: $N_{\text{pos}} = 12$. Value: `0.1498`.
* **Negative Class Brier ($BS_{\text{neg}}$):**
  $$BS_{\text{neg}} = \frac{1}{N_{\text{neg}}} \sum_{j \in \mathcal{N}} (\hat{p}_j - 0)^2$$
  * Denominator: $N_{\text{neg}} = 14$. Value: `0.1602`.

### 2.9 Expected Calibration Error (ECE)
* **Uniform-Bin ECE ($M = 10$ bins):**
  $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N_{\text{total}}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  * Denominator: $N_{\text{total}} = 26$. Value: `0.1621`.
* **Equal-Mass (Adaptive Quantile) ECE:**
  $$\text{ECE}_{\text{equal-mass}} = \sum_{k=1}^{K} \frac{1}{K} \left| \text{acc}(Q_k) - \text{conf}(Q_k) \right|$$
  * Bins defined by sample quantiles so each bin contains $\approx N/K$ records. Value: `0.1676`.

### 2.10 Temperature Scaling Calibration Parameter ($T^*$)
* **Definition:** Scalar temperature optimizing validation split log-likelihood:
  $$T^* = \arg\min_T -\sum_{i \in \text{val}} \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1-y_i) \log\left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$
* **Validation Denominator:** Evaluated strictly on the $N_{\text{val}} = 25$ validation records.
* **Leakage Control:** Fitted with zero test set exposure.
* **Optimal Value:** $T^* = 1.5649$.

### 2.11 Conformal Prediction and Abstention Metrics
* **Conformal Guarantee:** Split-conformal prediction sets $C(X) \subseteq \{0, 1\}$ satisfying $P(Y \in C(X)) \ge 1 - \epsilon$ at error level $\epsilon = 0.10$.
* **Abstention Definition:** A prediction abstains if the conformal prediction set contains both classes $C(X) = \{0, 1\}$ (clinical ambiguity) or is empty $C(X) = \emptyset$ (out-of-distribution), or if evidence reliability $R < 0.25$ and margin $< 0.10$.
* **Coverage (Answer Rate):**
  $$\text{Coverage} = \frac{N_{\text{retained}}}{N_{\text{total}}}$$
  * Denominator: $N_{\text{total}} = 26$.
* **Selective Accuracy (Retained Accuracy):**
  $$\text{Selective Accuracy} = \frac{1}{N_{\text{retained}}} \sum_{i \in \text{retained}} \mathbb{I}[\hat{y}_i = y_i]$$
  * Denominator: Retained predictions $N_{\text{retained}}$ (samples for which automated CDS decision is emitted).

---

## 3. Estimation Methods: Point Estimate vs. Bootstrap Mean vs. 95% CI

To guarantee complete scientific transparency, `reports/master_benchmark_results.json` explicitly decomposes every reported metric into three separate properties:

| Metric Representation | Mathematical Formulation | Interpretation |
| :--- | :--- | :--- |
| **Point Estimate (`point_estimates`)** | $\theta(\mathbf{y}_{\text{test}}, \hat{\mathbf{p}}_{\text{test}})$ | Empirical metric computed via `scikit-learn` on the fixed test partition ($N=26$). Reported headline number. |
| **Bootstrap Mean (`bootstrap_mean`)** | $\bar{\theta}_{\text{boot}} = \frac{1}{B} \sum_{b=1}^B \theta(\mathbf{y}^{(b)}, \hat{\mathbf{p}}^{(b)})$ | Expected metric value across $B = 500$ random resamples with replacement (seed = 42). |
| **95% Confidence Interval (`confidence_intervals_95`)** | $[\theta_{2.5\%}, \theta_{97.5\%}]$ | Empirical 2.5th and 97.5th percentiles across the 500 bootstrap replicates. |

> **Note on Finite-Sample Bootstrap Bias:** Because $N_{\text{test}} = 26$, small non-linear metric estimators (such as ROC-AUC and F1) exhibit slight sampling skewness: the bootstrap mean ($\text{ROC-AUC} \approx 0.8432$) can differ modestly from the sample point estimate ($\text{ROC-AUC} = 0.8512$). All headline claims correspond directly to the unrounded empirical point estimate.

---

## 4. Explanation of the Verification Audit Discrepancy

During an independent verification audit, an external verification script (`reports/verification/verify_step5_evaluation.py`) re-ran an independent training loop and observed:
* **Audit Script Output:** ROC-AUC `0.8393`, PR-AUC `0.8639`, F1 `0.7273`, Brier `0.1622`.
* **Benchmark Results JSON:** ROC-AUC `0.8512`, PR-AUC `0.8794`, F1 `0.7826`, Brier `0.1554`.

### Root Cause Analysis:
1. **Re-training Stochasticity vs. Stored Predictions:** The verification audit script did not evaluate previously saved test predictions; instead, it re-instantiated and re-trained a new neural network instance.
2. **Metadata Feature Dimensionality:** In `verify_step5_evaluation.py`, the metadata vector extracted via `summary.to_full_metadata_vector()` was 10-dimensional, missing the pharmacokinetics reachability padding (`rho_reach=0.5, f_m_CYP=0.5`) enforced in `run_verified_hdi_benchmark.py`. This subtle architectural mismatch altered the weights learned during training.
3. **Absence of Persisted Prediction Artifacts:** Prior to this revision, the benchmark script did not persist sample-level prediction vectors to disk, forcing auditors to re-implement model execution rather than auditing predictions directly.

### Corrective Implementation:
* `benchmarks/run_verified_hdi_benchmark.py` now explicitly persists:
  * `reports/benchmark_test_predictions.json` (all model predictions keyed by variant)
  * `reports/benchmark_test_predictions.csv` (tabular ground truth, predicted probability, and entity keys)
* `reports/master_benchmark_results.json` directly embeds `point_estimates`, `bootstrap_mean`, `confidence_intervals_95`, and sample-level `predictions`.
* `tests/test_metric_recomputation.py` executes `scikit-learn` directly on `benchmark_test_predictions.json` and asserts numerical equality down to $< 10^{-9}$ against `master_benchmark_results.json`.
