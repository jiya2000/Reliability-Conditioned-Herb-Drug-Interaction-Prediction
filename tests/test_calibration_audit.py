"""
Unit tests for calibration metrics: ECE, Brier score, and Temperature Scaling.

Tests required by Item 2:
(a) Constant predictor at the base rate (ECE must be < 0.03).
(b) Perfectly calibrated synthetic predictor.
(c) Overconfident synthetic predictor.
(d) Identification of non-identifiable temperature bounds.
"""

import numpy as np
import pytest
from scipy.special import expit, logit

from src.training.calibration import (
    compute_calibration_metrics,
    CalibrationMetrics,
)
from scripts.run_phase5_task_a import (
    compute_ece_equal_mass,
    fit_temperature_scaling,
)


def test_constant_predictor_at_base_rate():
    """
    Item 2 (a): Constant predictor at base rate must have ECE < 0.03.
    """
    n_pos = 76
    n_neg = 65
    y_true = np.array([1] * n_pos + [0] * n_neg)
    base_rate = float(n_pos) / len(y_true)
    y_prob = np.full(len(y_true), base_rate)

    ece_mass = compute_ece_equal_mass(y_true, y_prob, n_bins=10)
    assert ece_mass < 0.03, f"Constant predictor equal-mass ECE {ece_mass} >= 0.03"
    assert abs(ece_mass) < 1e-6, f"Constant predictor equal-mass ECE should be 0.0, got {ece_mass}"

    cal_metrics = compute_calibration_metrics(y_prob, y_true, num_bins=10)
    assert cal_metrics.expected_calibration_error < 0.03, (
        f"Equal-width ECE {cal_metrics.expected_calibration_error} >= 0.03"
    )
    assert cal_metrics.equal_mass_ece < 0.03, (
        f"Equal-mass ECE {cal_metrics.equal_mass_ece} >= 0.03"
    )

    # Expected Brier score is p*(1-p)
    expected_brier = base_rate * (1.0 - base_rate)
    assert abs(cal_metrics.brier_score - expected_brier) < 1e-4


def test_perfectly_calibrated_synthetic_predictor():
    """
    Item 2 (b): Perfectly calibrated synthetic predictor:
    p ~ Uniform(0.05, 0.95), y ~ Bernoulli(p).
    ECE should be < 0.05, and fitted temperature should be interior (approx 1.0).
    """
    rng = np.random.RandomState(42)
    n = 10000
    p = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(int)

    ece = compute_ece_equal_mass(y, p, n_bins=10)
    assert ece < 0.05, f"Perfectly calibrated predictor ECE {ece} >= 0.05"

    logits = logit(p)
    temp, ceiling_hit, status = fit_temperature_scaling(logits, y, bounds=(0.001, 1000.0))
    assert status == "identifiable", f"Expected identifiable temperature, got {status}"
    assert not ceiling_hit
    assert 0.85 <= temp <= 1.15, f"Expected temperature near 1.0, got {temp}"


def test_overconfident_synthetic_predictor():
    """
    Item 2 (c): Overconfident synthetic predictor:
    True logits z, predicted logits s = 3.0 * z.
    Uncalibrated ECE is high.
    Fitted temperature should be interior (~ 3.0).
    Calibrated ECE should drop below 0.05.
    """
    rng = np.random.RandomState(42)
    n = 10000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p_true).astype(int)

    z_true = logit(p_true)
    logits_overconfident = 3.0 * z_true
    p_overconfident = expit(logits_overconfident)

    ece_uncal = compute_ece_equal_mass(y, p_overconfident, n_bins=10)
    assert ece_uncal > 0.08, f"Overconfident predictor should have elevated ECE, got {ece_uncal}"

    temp, ceiling_hit, status = fit_temperature_scaling(logits_overconfident, y, bounds=(0.001, 1000.0))
    assert status == "identifiable", f"Expected identifiable temperature, got {status}"
    assert not ceiling_hit
    assert 2.5 <= temp <= 3.5, f"Expected temperature near 3.0, got {temp}"

    p_calibrated = expit(logits_overconfident / temp)
    ece_cal = compute_ece_equal_mass(y, p_calibrated, n_bins=10)
    assert ece_cal < 0.05, f"Calibrated ECE should be < 0.05, got {ece_cal}"
    assert ece_cal < ece_uncal, f"Calibration should improve ECE ({ece_cal} vs {ece_uncal})"


def test_non_identifiable_temperature_at_bound():
    """
    Item 2 (d): Non-identifiable logit distributions (e.g. overconfident chance-level logits)
    hit the boundary and must be flagged as 'not identifiable'.
    """
    rng = np.random.RandomState(42)
    n = 500
    # Overconfident random logits against 50/50 noise
    logits = rng.normal(0, 5.0, size=n)
    y = rng.choice([0, 1], size=n)

    temp, ceiling_hit, status = fit_temperature_scaling(logits, y, bounds=(0.001, 1000.0))
    assert status == "not identifiable", f"Expected not identifiable, got {status}"
    assert ceiling_hit
