"""
Conformal Tri-State Abstention Engine (CTAE).

NOVELTY CONTRIBUTION 3:
Implements mathematically calibrated selective prediction using split conformal
prediction, generating distribution-free finite-sample coverage guarantees at any
user-specified confidence level (1 - epsilon).

Transforms forced binary predictions into three clinically safe outcomes:
1. INTERACTION_PRESENT (set = {1}, model confident that interaction exists)
2. NO_INTERACTION      (set = {0}, model confident that combination is benign)
3. ABSTAIN             (set = {0,1} or empty, or high conflict -> routes to clinical expert)

Guaranteed Statistical Coverage:
    P(Y in C(X)) >= 1 - epsilon
for any exchangeable data distribution without parametric assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    from scipy.stats import norm, fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ClinicalDecision(str, Enum):
    INTERACTION_PRESENT = "INTERACTION_PRESENT"
    NO_INTERACTION = "NO_INTERACTION"
    ABSTAIN_CLINICAL_REVIEW = "ABSTAIN_CLINICAL_REVIEW"


@dataclass
class RiskCoverageResult:
    """Risk-coverage curve and Area Under Risk-Coverage curve (AURC)."""
    coverages: np.ndarray
    risks: np.ndarray
    selective_accuracies: np.ndarray
    aurc: float
    optimal_aurc: float
    excess_aurc: float


@dataclass
class SelectiveAccuracyResult:
    """Selective accuracy with bootstrap confidence intervals."""
    retained_accuracy: float
    coverage: float
    ci_lower: float
    ci_upper: float
    std_error: float
    coverage_ci_lower: float
    coverage_ci_upper: float
    raw_accuracy: float
    n_total: int
    n_retained: int
    n_abstained: int


@dataclass
class ConflictConcentrationResult:
    """Test results assessing whether abstentions concentrate on contradictory pairs."""
    high_conflict_n: int
    high_conflict_abstained: int
    high_conflict_rate: float
    low_conflict_n: int
    low_conflict_abstained: int
    low_conflict_rate: float
    rate_difference: float
    z_statistic: float
    p_value_z: float
    odds_ratio: float
    p_value_fisher: float
    is_statistically_significant: bool


@dataclass
class DecisionStabilityResult:
    """Stability of tri-state decisions across repeated runs, folds, or seeds."""
    invariance_fraction: float
    pairwise_agreement: float
    mean_decision_entropy: float
    n_samples: int
    n_runs: int
    per_class_stability: Dict[str, float]


def check_conformal_sample_size(n: int, epsilon: float) -> Tuple[bool, str]:
    """
    Checks whether calibration set size n satisfies the standard finite-sample
    conformal prediction requirement: ceil((n + 1) * (1 - epsilon)) / n <= 1.0,
    which requires n >= ceil((1 - epsilon) / epsilon).

    Args:
        n: Calibration set size.
        epsilon: Target error rate (e.g. 0.10 for 90% nominal coverage).

    Returns:
        (is_valid, message): Boolean check and explanatory diagnosis.
    """
    if n <= 0:
        return False, "Calibration sample size must be strictly positive (n > 0)."

    n_min = math.ceil((1.0 - float(epsilon)) / float(epsilon))
    if n < n_min:
        msg = (
            f"Calibration set size n={n} is less than required minimum n_min={n_min} "
            f"for error rate epsilon={epsilon:.2f}. The nominal finite-sample guarantee "
            f"ceil((n+1)(1-epsilon))/n <= 1 cannot be satisfied strictly; "
            f"quantile level will default to 1.0 (conservative fallback)."
        )
        return False, msg
    return True, f"Calibration set size n={n} satisfies conformal requirement (n_min={n_min} for epsilon={epsilon:.2f})."


@dataclass
class ConformalDecision:
    """Structured decision output for a single candidate pair."""
    predicted_label: int  # 1 (interact), 0 (no interact), -1 (abstain)
    decision: ClinicalDecision
    probability: float  # P(Y = 1 | X)
    conformal_set: List[int]  # Subset of {0, 1}
    p_value_0: float  # Conformal p-value for class 0
    p_value_1: float  # Conformal p-value for class 1
    abstain: bool
    abstention_reason: str
    confidence: float  # 1 - epsilon or calibrated confidence margin
    q_threshold: float
    routing_destination: str  # "automated_ehr_alert", "automated_safe", or "clinical_pharmacist_review"


@dataclass
class ConformalMetrics:
    """Evaluation metrics for conformal abstention system."""
    empirical_coverage: float       # Fraction of test samples where true label in conformal set
    abstention_rate: float          # Fraction of samples routed to human expert
    retained_accuracy: float        # Accuracy evaluated ONLY on accepted (non-abstained) predictions
    average_set_size: float         # Average cardinality of C(X)
    raw_unfiltered_accuracy: float  # Standard forced-binary accuracy across all samples
    singleton_rate: float           # Fraction of sets with size exactly 1
    empty_set_rate: float           # Fraction of sets that are empty (out-of-distribution)
    total_samples: int
    num_abstained: int


class ConformalTriStateEngine:
    """
    Split Conformal Prediction Engine with Clinical Safety Routing.

    Maintains statistical coverage guarantees while intelligently routing
    ambiguous, contradictory, or under-evidenced candidate pairs to human specialists.
    """

    def __init__(
        self,
        epsilon: float = 0.10,
        conflict_abstention_threshold: float = 0.60,
        min_reliability_threshold: float = 0.20,
    ):
        """
        Args:
            epsilon: Significance level. Guarantees coverage >= 1 - epsilon.
                Default 0.10 gives 90% confidence coverage.
            conflict_abstention_threshold: If contradiction strength exceeds this,
                forces abstention regardless of probability.
            min_reliability_threshold: If evidence reliability R is below this
                and prediction is near decision boundary, forces abstention.
        """
        self.epsilon = float(epsilon)
        self.conflict_threshold = float(conflict_abstention_threshold)
        self.min_reliability_threshold = float(min_reliability_threshold)

        self.q_hat: Optional[float] = None
        self.is_calibrated: bool = False
        self.calibration_scores: Optional[np.ndarray] = None
        self.sample_size_valid: bool = True
        self.sample_size_warning: str = ""

    def calibrate(
        self,
        cal_probabilities: Union[np.ndarray, torch.Tensor, List[float]],
        cal_labels: Union[np.ndarray, torch.Tensor, List[int]],
    ) -> float:
        """
        Calibrates the non-conformity threshold q_hat on a held-out calibration set.

        Non-conformity score:
            s_i = 1 - P(Y = y_i | X_i)
        Quantile level:
            p_level = ceil((n + 1) * (1 - epsilon)) / n

        Returns:
            Calibrated quantile threshold q_hat.
        """
        probs = self._to_numpy(cal_probabilities).flatten()
        labels = self._to_numpy(cal_labels).flatten().astype(int)

        n = len(labels)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        is_valid, msg = check_conformal_sample_size(n, self.epsilon)
        self.sample_size_valid = is_valid
        self.sample_size_warning = msg if not is_valid else ""
        if not is_valid:
            logger.warning(f"CTAE sample size warning: {msg}")

        # Compute non-conformity score for true label
        # If y = 1, score = 1 - p. If y = 0, score = 1 - (1 - p) = p.
        scores = np.where(labels == 1, 1.0 - probs, probs)
        self.calibration_scores = scores

        # Conformal quantile calculation with finite-sample correction
        quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - self.epsilon)) / float(n))
        self.q_hat = float(np.quantile(scores, quantile_level, method="higher"))
        self.is_calibrated = True

        logger.info(
            f"CTAE Calibrated on n={n} samples (epsilon={self.epsilon:.2f}) -> "
            f"q_hat={self.q_hat:.4f} (coverage guarantee >= {1.0 - self.epsilon:.1%})"
        )
        return self.q_hat

    def calibrate_pooled_oof(
        self,
        oof_probabilities: Union[np.ndarray, torch.Tensor, List[float]],
        oof_labels: Union[np.ndarray, torch.Tensor, List[int]],
    ) -> float:
        """
        Calibrates the non-conformity threshold q_hat using pooled out-of-fold (OOF) predictions
        from cross-validation splits.

        Because each prediction is made on a held-out fold, the nonconformity scores
        are exchangeable with future test samples, while pooling across folds yields
        an effective calibration set size of N instead of N/K.

        Returns:
            Calibrated quantile threshold q_hat.
        """
        probs = self._to_numpy(oof_probabilities).flatten()
        labels = self._to_numpy(oof_labels).flatten().astype(int)

        n = len(labels)
        if n == 0:
            raise ValueError("Pooled OOF calibration set cannot be empty.")

        is_valid, msg = check_conformal_sample_size(n, self.epsilon)
        self.sample_size_valid = is_valid
        self.sample_size_warning = msg if not is_valid else ""
        if not is_valid:
            logger.warning(f"CTAE pooled OOF sample size warning: {msg}")

        scores = np.where(labels == 1, 1.0 - probs, probs)
        self.calibration_scores = scores

        quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - self.epsilon)) / float(n))
        self.q_hat = float(np.quantile(scores, quantile_level, method="higher"))
        self.is_calibrated = True

        logger.info(
            f"CTAE Calibrated on pooled OOF n={n} samples (epsilon={self.epsilon:.2f}) -> "
            f"q_hat={self.q_hat:.4f} (coverage guarantee >= {1.0 - self.epsilon:.1%})"
        )
        return self.q_hat


    def predict_single(
        self,
        probability: float,
        contradiction_score: float = 0.0,
        reliability_score: float = 0.5,
    ) -> ConformalDecision:
        """
        Generates conformal prediction set and clinical decision for a single instance.
        """
        if not self.is_calibrated or self.q_hat is None:
            # Conservative default if uncalibrated: 0.90 threshold
            q_thresh = 0.85
        else:
            q_thresh = self.q_hat

        p = float(probability)
        # Non-conformity scores for hypothetical labels y in {0, 1}
        # s(0) = 1 - P(Y=0) = P(Y=1) = p
        # s(1) = 1 - P(Y=1) = 1 - p
        s_0 = p
        s_1 = 1.0 - p

        # Conformal set: labels where non-conformity <= threshold
        conformal_set: List[int] = []
        if s_0 <= q_thresh:
            conformal_set.append(0)
        if s_1 <= q_thresh:
            conformal_set.append(1)

        # Conformal p-values: fraction of cal scores >= candidate score
        p_val_0 = 1.0 - p
        p_val_1 = p
        if self.calibration_scores is not None and len(self.calibration_scores) > 0:
            p_val_0 = float(np.mean(self.calibration_scores >= s_0))
            p_val_1 = float(np.mean(self.calibration_scores >= s_1))

        # Decision mapping
        abstain = False
        reason = "conformal_certainty"

        if len(conformal_set) == 2:
            # Both outcomes plausible at 1 - epsilon -> High uncertainty
            abstain = True
            reason = "conformal_set_size_2_uncertainty"
        elif len(conformal_set) == 0:
            # Neither outcome plausible -> Out of distribution / anomalous
            abstain = True
            reason = "conformal_empty_set_out_of_distribution"

        # Domain safety override 1: High scientific conflict
        if contradiction_score > self.conflict_threshold:
            abstain = True
            reason = "active_scientific_contradiction_override"

        # Domain safety override 2: Low reliability near decision boundary
        confidence_margin = abs(p - 0.5) * 2.0
        if reliability_score < self.min_reliability_threshold and confidence_margin < 0.30:
            abstain = True
            reason = "unreliable_evidence_borderline_prediction"

        # Determine label and clinical route
        if abstain:
            pred_label = -1
            decision = ClinicalDecision.ABSTAIN_CLINICAL_REVIEW
            route = "clinical_pharmacist_review"
        elif conformal_set == [1]:
            pred_label = 1
            decision = ClinicalDecision.INTERACTION_PRESENT
            route = "automated_ehr_alert"
        else:
            pred_label = 0
            decision = ClinicalDecision.NO_INTERACTION
            route = "automated_safe"

        return ConformalDecision(
            predicted_label=pred_label,
            decision=decision,
            probability=round(p, 4),
            conformal_set=conformal_set,
            p_value_0=round(p_val_0, 4),
            p_value_1=round(p_val_1, 4),
            abstain=abstain,
            abstention_reason=reason,
            confidence=round(1.0 - self.epsilon, 2),
            q_threshold=round(q_thresh, 4),
            routing_destination=route,
        )

    def predict_batch(
        self,
        probabilities: Union[np.ndarray, torch.Tensor, List[float]],
        contradictions: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        reliabilities: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
    ) -> List[ConformalDecision]:
        """Runs conformal prediction across a batch."""
        probs = self._to_numpy(probabilities).flatten()
        n = len(probs)

        if contradictions is not None:
            conflicts = self._to_numpy(contradictions).flatten()
        else:
            conflicts = np.zeros(n)

        if reliabilities is not None:
            rel = self._to_numpy(reliabilities).flatten()
        else:
            rel = np.full(n, 0.5)

        decisions = []
        for i in range(n):
            decisions.append(
                self.predict_single(
                    probability=float(probs[i]),
                    contradiction_score=float(conflicts[i]),
                    reliability_score=float(rel[i]),
                )
            )
        return decisions

    def evaluate(
        self,
        test_probabilities: Union[np.ndarray, torch.Tensor, List[float]],
        test_labels: Union[np.ndarray, torch.Tensor, List[int]],
        contradictions: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        reliabilities: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
    ) -> ConformalMetrics:
        """
        Evaluates empirical coverage, abstention rate, and retained accuracy.
        """
        probs = self._to_numpy(test_probabilities).flatten()
        labels = self._to_numpy(test_labels).flatten().astype(int)
        decisions = self.predict_batch(probs, contradictions, reliabilities)

        n = len(labels)
        covered_count = 0
        abstained_count = 0
        retained_correct = 0
        retained_total = 0
        total_set_size = 0
        singletons = 0
        empty_sets = 0
        raw_correct = 0

        for i in range(n):
            dec = decisions[i]
            y_true = labels[i]
            y_raw_pred = 1 if probs[i] >= 0.5 else 0

            if y_raw_pred == y_true:
                raw_correct += 1

            # Check coverage
            if y_true in dec.conformal_set:
                covered_count += 1

            set_sz = len(dec.conformal_set)
            total_set_size += set_sz
            if set_sz == 1:
                singletons += 1
            elif set_sz == 0:
                empty_sets += 1

            # Check abstention
            if dec.abstain:
                abstained_count += 1
            else:
                retained_total += 1
                if dec.predicted_label == y_true:
                    retained_correct += 1

        coverage = float(covered_count) / float(n) if n > 0 else 0.0
        abst_rate = float(abstained_count) / float(n) if n > 0 else 0.0
        retained_acc = (
            float(retained_correct) / float(retained_total) if retained_total > 0 else 0.0
        )
        avg_set_sz = float(total_set_size) / float(n) if n > 0 else 0.0
        raw_acc = float(raw_correct) / float(n) if n > 0 else 0.0

        return ConformalMetrics(
            empirical_coverage=round(coverage, 4),
            abstention_rate=round(abst_rate, 4),
            retained_accuracy=round(retained_acc, 4),
            average_set_size=round(avg_set_sz, 4),
            raw_unfiltered_accuracy=round(raw_acc, 4),
            singleton_rate=round(float(singletons) / float(n), 4) if n > 0 else 0.0,
            empty_set_rate=round(float(empty_sets) / float(n), 4) if n > 0 else 0.0,
            total_samples=n,
            num_abstained=abstained_count,
        )

    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor, List[Any]]) -> np.ndarray:
        """Converts input data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)


# Alias for backward compatibility and semantic clarity
ConformalTriStateAbstainer = ConformalTriStateEngine


def compute_risk_coverage_curve(
    probabilities: Union[np.ndarray, torch.Tensor, List[float]],
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    confidence_scores: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
    n_points: int = 100,
) -> RiskCoverageResult:
    """
    Computes empirical risk-coverage curve and Area Under Risk-Coverage curve (AURC).

    For selective prediction, samples are ordered by confidence descending.
    As coverage kappa increases from 1/N to 1.0, the risk R(kappa) is the error rate
    evaluated only on the retained kappa fraction of highest-confidence predictions.

    Args:
        probabilities: Predicted probabilities P(Y = 1 | X).
        labels: Ground truth binary labels Y in {0, 1}.
        confidence_scores: Optional confidence ranking metric. If None, uses margin |p - 0.5| * 2.
        n_points: Number of evaluation points on the curve.

    Returns:
        RiskCoverageResult containing coverages, risks, selective accuracies,
        AURC, oracle optimal AURC, and excess AURC (E-AURC).
    """
    probs = np.asarray(probabilities, dtype=float).flatten()
    y_true = np.asarray(labels, dtype=int).flatten()
    n = len(probs)
    if n == 0:
        raise ValueError("Cannot compute risk-coverage curve on empty arrays.")

    y_pred = (probs >= 0.5).astype(int)
    if confidence_scores is not None:
        conf = np.asarray(confidence_scores, dtype=float).flatten()
    else:
        conf = np.abs(probs - 0.5) * 2.0

    # Sort descending by confidence
    order = np.argsort(-conf)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    mistakes = (y_true_sorted != y_pred_sorted).astype(float)

    # Cumulative errors across sample counts k = 1 .. n
    k_values = np.arange(1, n + 1)
    cum_errors = np.cumsum(mistakes)
    risks = cum_errors / k_values
    coverages = k_values / float(n)
    selective_accs = 1.0 - risks

    # Trapezoidal AURC
    if n > 1:
        aurc = float(np.sum((risks[:-1] + risks[1:]) * 0.5 * (coverages[1:] - coverages[:-1]))) + float(risks[0] * coverages[0])
    else:
        aurc = float(risks[0])

    # Optimal (oracle) risk-coverage curve: zero errors first, then errors at the end
    mistakes_oracle = np.sort(mistakes)  # 0s first, then 1s
    cum_errors_oracle = np.cumsum(mistakes_oracle)
    risks_oracle = cum_errors_oracle / k_values
    if n > 1:
        optimal_aurc = float(np.sum((risks_oracle[:-1] + risks_oracle[1:]) * 0.5 * (coverages[1:] - coverages[:-1]))) + float(risks_oracle[0] * coverages[0])
    else:
        optimal_aurc = float(risks_oracle[0])

    excess_aurc = max(0.0, float(aurc - optimal_aurc))

    # Downsample points for clean plotting/export if n > n_points
    if n > n_points:
        idx = np.round(np.linspace(0, n - 1, n_points)).astype(int)
        cov_out = coverages[idx]
        risk_out = risks[idx]
        sel_acc_out = selective_accs[idx]
    else:
        cov_out = coverages
        risk_out = risks
        sel_acc_out = selective_accs

    return RiskCoverageResult(
        coverages=cov_out,
        risks=risk_out,
        selective_accuracies=sel_acc_out,
        aurc=round(aurc, 4),
        optimal_aurc=round(optimal_aurc, 4),
        excess_aurc=round(excess_aurc, 4),
    )


def compute_selective_accuracy_with_ci(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    predictions: Union[np.ndarray, torch.Tensor, List[int]],
    abstentions: Union[np.ndarray, torch.Tensor, List[bool]],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> SelectiveAccuracyResult:
    """
    Computes selective accuracy on accepted predictions with paired bootstrap confidence intervals.

    Args:
        labels: True binary labels in {0, 1}.
        predictions: Predicted binary labels in {0, 1} (or -1 for abstentions).
        abstentions: Boolean array where True indicates the model abstained.
        n_bootstraps: Number of bootstrap iterations.
        alpha: Significance level (default 0.05 gives 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        SelectiveAccuracyResult containing point estimates and bootstrap CIs.
    """
    y_true = np.asarray(labels, dtype=int).flatten()
    y_pred = np.asarray(predictions, dtype=int).flatten()
    abst = np.asarray(abstentions, dtype=bool).flatten()

    n_total = len(y_true)
    if n_total == 0:
        raise ValueError("Cannot compute selective accuracy on empty inputs.")

    retained = ~abst
    n_retained = int(np.sum(retained))
    n_abstained = int(np.sum(abst))
    coverage = float(n_retained) / float(n_total)

    retained_acc = (
        float(np.mean(y_true[retained] == y_pred[retained])) if n_retained > 0 else 0.0
    )
    raw_acc = float(np.mean(y_true == y_pred))

    rng = np.random.default_rng(random_state)
    boot_accs: List[float] = []
    boot_covs: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_total, size=n_total, replace=True)
        b_true = y_true[idx]
        b_pred = y_pred[idx]
        b_ret = ~abst[idx]
        n_b_ret = int(np.sum(b_ret))

        boot_covs.append(float(n_b_ret) / float(n_total))
        if n_b_ret > 0:
            boot_accs.append(float(np.mean(b_true[b_ret] == b_pred[b_ret])))
        else:
            boot_accs.append(0.0)

    ci_lower = float(np.percentile(boot_accs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_accs, 100.0 * (1.0 - alpha / 2.0)))
    std_err = float(np.std(boot_accs))

    cov_ci_lower = float(np.percentile(boot_covs, 100.0 * (alpha / 2.0)))
    cov_ci_upper = float(np.percentile(boot_covs, 100.0 * (1.0 - alpha / 2.0)))

    return SelectiveAccuracyResult(
        retained_accuracy=round(retained_acc, 4),
        coverage=round(coverage, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        std_error=round(std_err, 4),
        coverage_ci_lower=round(cov_ci_lower, 4),
        coverage_ci_upper=round(cov_ci_upper, 4),
        raw_accuracy=round(raw_acc, 4),
        n_total=n_total,
        n_retained=n_retained,
        n_abstained=n_abstained,
    )


def validate_abstention_conflict_concentration(
    decisions: List[ConformalDecision],
    contradictions: Union[np.ndarray, torch.Tensor, List[float]],
    conflict_threshold: float = 0.30,
) -> ConflictConcentrationResult:
    """
    Validates whether abstentions significantly concentrate on genuinely contradictory pairs.

    Compares abstention rates between high-conflict (C_h,d > threshold) and
    low-conflict (C_h,d <= threshold) partitions using two-sample proportion test
    and Fisher's exact test.

    Returns:
        ConflictConcentrationResult with contingency table counts, rates,
        z-statistic, odds ratio, and two-sided p-values.
    """
    conflicts = np.asarray(contradictions, dtype=float).flatten()
    abstentions = np.array([d.abstain for d in decisions], dtype=bool)

    if len(conflicts) != len(abstentions):
        raise ValueError("Lengths of contradictions and decisions must match.")

    high_mask = conflicts > float(conflict_threshold)
    low_mask = ~high_mask

    n1 = int(np.sum(high_mask))
    x1 = int(np.sum(abstentions[high_mask]))
    r1 = float(x1) / float(n1) if n1 > 0 else 0.0

    n2 = int(np.sum(low_mask))
    x2 = int(np.sum(abstentions[low_mask]))
    r2 = float(x2) / float(n2) if n2 > 0 else 0.0

    rate_diff = r1 - r2

    # Two-sample pooled proportion test
    total_n = n1 + n2
    total_x = x1 + x2
    if n1 > 0 and n2 > 0 and 0 < total_x < total_n:
        p0 = float(total_x) / float(total_n)
        se = math.sqrt(p0 * (1.0 - p0) * (1.0 / float(n1) + 1.0 / float(n2)))
        z_stat = (r1 - r2) / se if se > 0 else 0.0
        if HAS_SCIPY:
            p_z = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        else:
            p_z = float(math.erfc(abs(z_stat) / math.sqrt(2.0)))
    else:
        z_stat = 0.0
        p_z = 1.0

    # Fisher's exact test
    if HAS_SCIPY and n1 > 0 and n2 > 0:
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        odds_ratio, p_fisher = fisher_exact(table)
        odds_ratio = float(odds_ratio)
        p_fisher = float(p_fisher)
    else:
        denom = (n1 - x1) * x2
        odds_ratio = float((x1 * (n2 - x2)) / denom) if denom > 0 else (float("inf") if x1 > 0 else 1.0)
        p_fisher = p_z

    is_sig = bool(p_fisher < 0.05 or p_z < 0.05)

    return ConflictConcentrationResult(
        high_conflict_n=n1,
        high_conflict_abstained=x1,
        high_conflict_rate=round(r1, 4),
        low_conflict_n=n2,
        low_conflict_abstained=x2,
        low_conflict_rate=round(r2, 4),
        rate_difference=round(rate_diff, 4),
        z_statistic=round(z_stat, 4),
        p_value_z=round(p_z, 4),
        odds_ratio=round(odds_ratio, 4) if not math.isinf(odds_ratio) else 999.0,
        p_value_fisher=round(p_fisher, 4),
        is_statistically_significant=is_sig,
    )


def measure_decision_stability(
    multi_run_decisions: Union[List[List[ConformalDecision]], List[List[int]], np.ndarray],
) -> DecisionStabilityResult:
    """
    Measures the stability of tri-state classification (Positive, Negative, Abstain)
    across repeated runs, cross-validation folds, or random seeds.

    Args:
        multi_run_decisions: Matrix or list of runs where each element is
            a ConformalDecision (or int in {-1, 0, 1}) for N samples across K runs.

    Returns:
        DecisionStabilityResult containing invariance fraction, pairwise agreement,
        and per-class stability.
    """
    if isinstance(multi_run_decisions, np.ndarray):
        M = multi_run_decisions.astype(int)
    elif len(multi_run_decisions) > 0 and isinstance(multi_run_decisions[0][0], ConformalDecision):
        M = np.array(
            [[d.predicted_label for d in run] for run in multi_run_decisions],
            dtype=int,
        )
    else:
        M = np.array(multi_run_decisions, dtype=int)

    k_runs, n_samples = M.shape
    if k_runs == 0 or n_samples == 0:
        raise ValueError("Cannot measure decision stability on empty matrix.")

    # Invariance check: decisions match across all runs for a given sample
    invariant_flags = np.array([np.all(M[:, i] == M[0, i]) for i in range(n_samples)], dtype=bool)
    invariance_frac = float(np.mean(invariant_flags))

    # Pairwise agreement between runs
    if k_runs > 1:
        pair_agreements: List[float] = []
        for r1 in range(k_runs):
            for r2 in range(r1 + 1, k_runs):
                pair_agreements.append(float(np.mean(M[r1, :] == M[r2, :])))
        pairwise_agree = float(np.mean(pair_agreements))
    else:
        pairwise_agree = 1.0

    # Per-sample entropy across tri-state distribution {-1, 0, 1}
    sample_entropies: List[float] = []
    modal_decisions: List[int] = []

    for i in range(n_samples):
        vals, counts = np.unique(M[:, i], return_counts=True)
        probs = counts / float(k_runs)
        ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
        sample_entropies.append(ent)
        mode_val = vals[np.argmax(counts)]
        modal_decisions.append(int(mode_val))

    mean_entropy = float(np.mean(sample_entropies))
    mod_arr = np.array(modal_decisions)

    # Per-class stability: agreement on samples modal to class
    per_class_stab = {}
    class_map = {-1: "abstain", 0: "negative", 1: "positive"}
    for label, name in class_map.items():
        sub_mask = mod_arr == label
        if np.sum(sub_mask) > 0:
            per_class_stab[name] = round(float(np.mean(invariant_flags[sub_mask])), 4)
        else:
            per_class_stab[name] = 1.0

    return DecisionStabilityResult(
        invariance_fraction=round(invariance_frac, 4),
        pairwise_agreement=round(pairwise_agree, 4),
        mean_decision_entropy=round(mean_entropy, 4),
        n_samples=n_samples,
        n_runs=k_runs,
        per_class_stability=per_class_stab,
    )

