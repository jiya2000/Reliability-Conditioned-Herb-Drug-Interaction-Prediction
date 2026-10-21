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



class ClinicalDecision(str, Enum):
    INTERACTION_PRESENT = "INTERACTION_PRESENT"
    NO_INTERACTION = "NO_INTERACTION"
    ABSTAIN_CLINICAL_REVIEW = "ABSTAIN_CLINICAL_REVIEW"


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
