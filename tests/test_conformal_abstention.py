"""
Tests for Novelty 3: Conformal Tri-State Abstention Engine (CTAE).
"""

import numpy as np
import pytest
import torch
from src.models.conformal_abstention import (
    ConformalTriStateEngine,
    ClinicalDecision,
    ConformalDecision,
    ConformalMetrics,
)


def test_conformal_calibration_guarantee():
    """
    Test split conformal calibration on synthetic predictions.
    With epsilon = 0.10, empirical coverage on test set should approach >= 90%.
    """
    np.random.seed(42)
    engine = ConformalTriStateEngine(epsilon=0.10)

    # Synthetic calibration set: 200 samples
    n_cal = 200
    y_cal = np.random.binomial(1, 0.5, size=n_cal)
    # Generate calibrated probabilities with reasonable accuracy
    probs_cal = np.where(y_cal == 1, np.random.uniform(0.6, 0.95, size=n_cal), np.random.uniform(0.05, 0.4, size=n_cal))

    q_hat = engine.calibrate(probs_cal, y_cal)
    assert engine.is_calibrated
    assert 0.0 < q_hat < 1.0

    # Synthetic test set: 500 samples
    n_test = 500
    y_test = np.random.binomial(1, 0.5, size=n_test)
    probs_test = np.where(y_test == 1, np.random.uniform(0.6, 0.95, size=n_test), np.random.uniform(0.05, 0.4, size=n_test))

    metrics = engine.evaluate(probs_test, y_test)

    # Coverage should meet or exceed 1 - epsilon (within statistical margin)
    assert metrics.empirical_coverage >= 0.88
    # Retained accuracy on accepted predictions should be at least as high as unfiltered
    assert metrics.retained_accuracy >= metrics.raw_unfiltered_accuracy

    # Also test with noisy predictions near boundary where abstention improves accuracy
    probs_noisy = probs_test.copy()
    noise_idx = np.random.choice(n_test, size=50, replace=False)
    probs_noisy[noise_idx] = 0.50  # ambiguous borderline cases
    metrics_noisy = engine.evaluate(probs_noisy, y_test)
    assert metrics_noisy.abstention_rate > 0.0



def test_conformal_tri_state_decisions():
    engine = ConformalTriStateEngine(epsilon=0.10, conflict_abstention_threshold=0.60)
    # Set q_hat manually for deterministic unit testing
    engine.q_hat = 0.40
    engine.is_calibrated = True

    # 1. Very confident positive: P = 0.95
    # s(1) = 0.05 <= 0.40 (in set), s(0) = 0.95 > 0.40 (not in set) -> {1}
    dec_pos = engine.predict_single(probability=0.95)
    assert dec_pos.predicted_label == 1
    assert dec_pos.decision == ClinicalDecision.INTERACTION_PRESENT
    assert not dec_pos.abstain
    assert dec_pos.conformal_set == [1]

    # 2. Very confident negative: P = 0.05
    # s(0) = 0.05 <= 0.40 (in set), s(1) = 0.95 > 0.40 (not in set) -> {0}
    dec_neg = engine.predict_single(probability=0.05)
    assert dec_neg.predicted_label == 0
    assert dec_neg.decision == ClinicalDecision.NO_INTERACTION
    assert not dec_neg.abstain
    assert dec_neg.conformal_set == [0]

    # 3. Borderline ambiguous: P = 0.50
    # s(0) = 0.50 > 0.40 and s(1) = 0.50 > 0.40 -> set is empty or {0, 1} depending on q
    # With q_hat = 0.60, s(0)=0.50 <= 0.60, s(1)=0.50 <= 0.60 -> set is {0, 1} -> ABSTAIN
    engine.q_hat = 0.60
    dec_ambiguous = engine.predict_single(probability=0.50)
    assert dec_ambiguous.predicted_label == -1
    assert dec_ambiguous.decision == ClinicalDecision.ABSTAIN_CLINICAL_REVIEW
    assert dec_ambiguous.abstain
    assert dec_ambiguous.conformal_set == [0, 1]


def test_conflict_override_abstention():
    engine = ConformalTriStateEngine(epsilon=0.10, conflict_abstention_threshold=0.60)
    engine.q_hat = 0.40
    engine.is_calibrated = True

    # Even with high probability (0.90), high contradiction (> 0.60) forces abstention!
    dec = engine.predict_single(probability=0.90, contradiction_score=0.75)
    assert dec.abstain
    assert dec.predicted_label == -1
    assert dec.decision == ClinicalDecision.ABSTAIN_CLINICAL_REVIEW
    assert "contradiction" in dec.abstention_reason
