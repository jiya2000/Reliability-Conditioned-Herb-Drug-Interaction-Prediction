"""
Unit tests for Phase 7: Abstention and Conformal Evaluation.
Tests:
1. Conformal assumption and sample size checks.
2. Split-conformal calibration with pooled out-of-fold predictions.
3. Risk-coverage curves and Area Under Risk-Coverage (AURC).
4. Selective accuracy with bootstrap confidence intervals.
5. Conflict concentration validation (z-test and Fisher's exact test).
6. Tri-state decision stability across repeated runs/folds.
"""

import math
import numpy as np
import pytest

from src.models.conformal_abstention import (
    ConformalTriStateEngine,
    ClinicalDecision,
    ConformalDecision,
    check_conformal_sample_size,
    compute_risk_coverage_curve,
    compute_selective_accuracy_with_ci,
    validate_abstention_conflict_concentration,
    measure_decision_stability,
    RiskCoverageResult,
    SelectiveAccuracyResult,
    ConflictConcentrationResult,
    DecisionStabilityResult,
)


def test_conformal_sample_size_checks():
    """Verify finite-sample conformal requirement n >= ceil((1-eps)/eps)."""
    # For epsilon = 0.10: n_min = ceil(0.90 / 0.10) = 9
    is_valid, msg = check_conformal_sample_size(n=5, epsilon=0.10)
    assert not is_valid
    assert "required minimum n_min=9" in msg

    is_valid_ok, msg_ok = check_conformal_sample_size(n=10, epsilon=0.10)
    assert is_valid_ok
    assert "satisfies" in msg_ok

    # For epsilon = 0.05: n_min = ceil(0.95 / 0.05) = 19
    is_valid_5, _ = check_conformal_sample_size(n=15, epsilon=0.05)
    assert not is_valid_5

    is_valid_20, _ = check_conformal_sample_size(n=20, epsilon=0.05)
    assert is_valid_20

    # Non-positive sample size
    is_valid_zero, _ = check_conformal_sample_size(n=0, epsilon=0.10)
    assert not is_valid_zero


def test_conformal_calibrate_pooled_oof():
    """Verify pooled out-of-fold calibration produces valid threshold and coverage."""
    rng = np.random.default_rng(42)
    engine = ConformalTriStateEngine(epsilon=0.10)

    # Simulated pooled out-of-fold predictions across 5 folds (N = 100)
    y_oof = rng.binomial(1, 0.5, size=100)
    probs_oof = np.where(y_oof == 1, rng.uniform(0.65, 0.95, size=100), rng.uniform(0.05, 0.35, size=100))

    q_hat = engine.calibrate_pooled_oof(probs_oof, y_oof)
    assert engine.is_calibrated
    assert 0.0 < q_hat < 1.0
    assert engine.sample_size_valid

    # Evaluate on held-out test set
    y_test = rng.binomial(1, 0.5, size=200)
    probs_test = np.where(y_test == 1, rng.uniform(0.65, 0.95, size=200), rng.uniform(0.05, 0.35, size=200))
    metrics = engine.evaluate(probs_test, y_test)

    assert metrics.empirical_coverage >= 0.88
    assert metrics.total_samples == 200


def test_risk_coverage_curve_and_aurc():
    """Verify risk-coverage curve, monotonicity of errors, and AURC bounds."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.binomial(1, 0.5, size=n)
    # Generate probabilities where high confidence correlates with correctness
    probs = np.where(y_true == 1, rng.uniform(0.60, 0.98, size=n), rng.uniform(0.02, 0.40, size=n))
    # Inject 10 borderline ambiguous mistakes
    flip_idx = rng.choice(n, size=10, replace=False)
    probs[flip_idx] = 0.51  # borderline mistakes

    res = compute_risk_coverage_curve(probs, y_true, n_points=50)

    assert isinstance(res, RiskCoverageResult)
    assert len(res.coverages) == 50
    assert res.coverages[-1] == 1.0
    # Risk at 100% coverage should match overall raw error rate
    y_pred = (probs >= 0.5).astype(int)
    raw_error_rate = float(np.mean(y_true != y_pred))
    assert math.isclose(res.risks[-1], raw_error_rate, abs_tol=1e-3)

    # AURC properties
    assert 0.0 <= res.optimal_aurc <= res.aurc <= 1.0
    assert res.excess_aurc >= 0.0


def test_selective_accuracy_with_bootstrap_ci():
    """Verify selective accuracy calculation and paired bootstrap confidence intervals."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.binomial(1, 0.5, size=n)
    y_pred = y_true.copy()
    # Introduce 20 mistakes
    mistake_idx = rng.choice(n, size=20, replace=False)
    y_pred[mistake_idx] = 1 - y_true[mistake_idx]

    # Abstain on 15 of the 20 mistakes
    abstentions = np.zeros(n, dtype=bool)
    abstentions[mistake_idx[:15]] = True

    res = compute_selective_accuracy_with_ci(
        y_true, y_pred, abstentions, n_bootstraps=500, alpha=0.05, random_state=42
    )

    assert isinstance(res, SelectiveAccuracyResult)
    assert res.n_total == 100
    assert res.n_abstained == 15
    assert res.n_retained == 85
    assert math.isclose(res.coverage, 0.85, abs_tol=1e-3)

    # Retained accuracy must be strictly higher than raw accuracy because abstentions removed mistakes
    assert res.retained_accuracy > res.raw_accuracy
    # Confidence interval must bound point estimate
    assert res.ci_lower <= res.retained_accuracy <= res.ci_upper
    assert res.coverage_ci_lower <= res.coverage <= res.coverage_ci_upper


def test_validate_abstention_conflict_concentration():
    """Verify statistical test detecting concentration of abstentions on contradictory pairs."""
    engine = ConformalTriStateEngine(epsilon=0.10, conflict_abstention_threshold=0.30)
    engine.q_hat = 0.40
    engine.is_calibrated = True

    # 50 high-conflict pairs (C = 0.70) and 50 low-conflict pairs (C = 0.05)
    conflicts = np.array([0.70] * 50 + [0.05] * 50)
    probs = np.array([0.85] * 100)  # confident probability

    decisions = [
        engine.predict_single(probability=float(probs[i]), contradiction_score=float(conflicts[i]))
        for i in range(100)
    ]

    res = validate_abstention_conflict_concentration(decisions, conflicts, conflict_threshold=0.30)

    assert isinstance(res, ConflictConcentrationResult)
    assert res.high_conflict_n == 50
    assert res.low_conflict_n == 50
    # Because contradiction > 0.30 triggers override, all high-conflict pairs should abstain
    assert res.high_conflict_abstained == 50
    assert res.high_conflict_rate == 1.0
    assert res.low_conflict_abstained == 0
    assert res.low_conflict_rate == 0.0
    assert res.rate_difference == 1.0

    # Must be overwhelmingly statistically significant
    assert res.is_statistically_significant
    assert res.p_value_fisher < 0.001
    assert res.p_value_z < 0.001


def test_decision_stability_across_runs():
    """Verify decision stability measurement across repeated runs."""
    # Case 1: Perfectly invariant decisions across 5 runs for 20 samples
    M_perfect = np.tile(np.array([1, 0, -1, 1, 0] * 4), (5, 1))
    res_perfect = measure_decision_stability(M_perfect)

    assert res_perfect.invariance_fraction == 1.0
    assert res_perfect.pairwise_agreement == 1.0
    assert res_perfect.mean_decision_entropy == 0.0
    assert res_perfect.per_class_stability["positive"] == 1.0
    assert res_perfect.per_class_stability["negative"] == 1.0
    assert res_perfect.per_class_stability["abstain"] == 1.0

    # Case 2: Flapping decisions across runs
    M_flapping = np.array([
        [1, 0, -1, 1],
        [-1, 0, -1, 0],
        [1, 0, -1, -1],
    ])
    res_flapping = measure_decision_stability(M_flapping)

    assert res_flapping.invariance_fraction < 1.0
    assert res_flapping.pairwise_agreement < 1.0
    assert res_flapping.mean_decision_entropy > 0.0
    # Sample index 1 (0, 0, 0) and 2 (-1, -1, -1) are invariant
    assert math.isclose(res_flapping.invariance_fraction, 0.50, abs_tol=1e-3)
