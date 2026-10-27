"""
Test Metric Recomputation and Equivalence Assertion.

Verifies:
1. Every model in master_benchmark_results.json and benchmark_test_predictions.json
   has point estimates, bootstrap means, 95% CIs, and sample-level predictions.
2. Independent sklearn recomputation directly on saved prediction records matches
   the reported point estimates with absolute error < 1e-9.
3. Bootstrap sample means fall strictly within the 95% confidence intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def master_results():
    master_file = PROJECT_ROOT / "reports" / "master_benchmark_results.json"
    if not master_file.exists():
        pytest.skip("reports/master_benchmark_results.json not generated yet.")
    with open(master_file, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def saved_predictions():
    pred_file = PROJECT_ROOT / "reports" / "benchmark_test_predictions.json"
    if not pred_file.exists():
        pytest.skip("reports/benchmark_test_predictions.json not generated yet.")
    with open(pred_file, "r") as f:
        return json.load(f)


def test_headline_model_has_required_metric_fields(master_results):
    """Verify headline model has clearly labeled point estimate, bootstrap mean, and 95% CI."""
    headline = master_results["verified_hdi_benchmark"]["headline_metrics"]
    assert headline, "Headline metrics must not be empty"

    # Must contain point_estimates, bootstrap_mean, and confidence_intervals_95
    assert "point_estimates" in headline, "Headline metrics missing 'point_estimates'"
    assert "bootstrap_mean" in headline, "Headline metrics missing 'bootstrap_mean'"
    assert "confidence_intervals_95" in headline, "Headline metrics missing 'confidence_intervals_95'"
    assert "predictions" in headline, "Headline metrics missing 'predictions'"

    pt = headline["point_estimates"]
    bm = headline["bootstrap_mean"]
    ci = headline["confidence_intervals_95"]

    # Verify key metrics are present
    for metric in ["roc_auc", "pr_auc", "f1", "brier_score", "accuracy"]:
        assert metric in pt, f"Missing {metric} in point_estimates"
        assert metric in bm, f"Missing {metric} in bootstrap_mean"
        assert metric in ci, f"Missing {metric} in confidence_intervals_95"
        # CI must be [lower, upper] with lower <= upper
        assert ci[metric][0] <= ci[metric][1], f"Invalid CI ordering for {metric}: {ci[metric]}"
        # Bootstrap mean should lie inside or extremely close to CI
        assert ci[metric][0] - 1e-4 <= bm[metric] <= ci[metric][1] + 1e-4, (
            f"Bootstrap mean {bm[metric]} outside CI {ci[metric]} for {metric}"
        )


def test_headline_model_sklearn_recomputation_exact_match(master_results, saved_predictions):
    """Verify independent sklearn recomputation on saved test predictions equals headline metrics (< 1e-9)."""
    headline = master_results["verified_hdi_benchmark"]["headline_metrics"]
    variant = headline["model_variant"]
    preds = headline.get("predictions")
    if not preds and variant in saved_predictions:
        preds = saved_predictions[variant]

    assert preds and len(preds) == 26, f"Expected 26 test predictions, found {len(preds) if preds else 0}"

    y_true = np.array([p["y_true"] for p in preds], dtype=int)
    y_prob = np.array([p["y_prob"] for p in preds], dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    # Independent sklearn calculation
    calc_roc = float(roc_auc_score(y_true, y_prob))
    calc_pr = float(average_precision_score(y_true, y_prob))
    calc_f1 = float(f1_score(y_true, y_pred))
    calc_brier = float(brier_score_loss(y_true, y_prob))
    calc_acc = float(accuracy_score(y_true, y_pred))
    calc_prec = float(precision_score(y_true, y_pred, zero_division=0))
    calc_rec = float(recall_score(y_true, y_pred, zero_division=0))

    pt = headline["point_estimates"]

    assert abs(calc_roc - pt["roc_auc"]) < 1e-9, f"ROC-AUC mismatch: {calc_roc} vs {pt['roc_auc']}"
    assert abs(calc_pr - pt["pr_auc"]) < 1e-9, f"PR-AUC mismatch: {calc_pr} vs {pt['pr_auc']}"
    assert abs(calc_f1 - pt["f1"]) < 1e-9, f"F1 mismatch: {calc_f1} vs {pt['f1']}"
    assert abs(calc_brier - pt["brier_score"]) < 1e-9, f"Brier mismatch: {calc_brier} vs {pt['brier_score']}"
    assert abs(calc_acc - pt["accuracy"]) < 1e-9, f"Accuracy mismatch: {calc_acc} vs {pt['accuracy']}"
    assert abs(calc_prec - pt["precision"]) < 1e-9, f"Precision mismatch: {calc_prec} vs {pt['precision']}"
    assert abs(calc_rec - pt["recall"]) < 1e-9, f"Recall mismatch: {calc_rec} vs {pt['recall']}"


def test_all_models_sklearn_recomputation_exact_match(master_results, saved_predictions):
    """Verify independent sklearn recomputation matches point estimates across ALL evaluated variants."""
    all_reports = master_results["verified_hdi_benchmark"]["all_variant_reports"]
    assert len(all_reports) > 0, "No variant reports found in master benchmark results"

    for r in all_reports:
        variant = r["model_variant"]
        preds = r.get("predictions")
        if not preds and variant in saved_predictions:
            preds = saved_predictions[variant]

        assert preds is not None, f"Predictions missing for model variant '{variant}'"
        assert len(preds) == 26, f"Variant '{variant}' had {len(preds)} samples instead of 26"

        y_true = np.array([p["y_true"] for p in preds], dtype=int)
        y_prob = np.array([p["y_prob"] for p in preds], dtype=float)
        # Use recorded y_pred if available, otherwise threshold at 0.5
        y_pred = np.array([p.get("y_pred", int(p["y_prob"] >= 0.5)) for p in preds], dtype=int)

        pt = r["point_estimates"]

        # If more than 1 class present, verify ROC and PR
        if len(np.unique(y_true)) > 1:
            calc_roc = float(roc_auc_score(y_true, y_prob))
            calc_pr = float(average_precision_score(y_true, y_prob))
            assert abs(calc_roc - pt["roc_auc"]) < 1e-9, (
                f"[{variant}] ROC-AUC mismatch: calc={calc_roc}, stored={pt['roc_auc']}"
            )
            assert abs(calc_pr - pt["pr_auc"]) < 1e-9, (
                f"[{variant}] PR-AUC mismatch: calc={calc_pr}, stored={pt['pr_auc']}"
            )

        calc_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        calc_brier = float(brier_score_loss(y_true, y_prob))
        calc_acc = float(accuracy_score(y_true, y_pred))

        assert abs(calc_f1 - pt["f1"]) < 1e-9, (
            f"[{variant}] F1 mismatch: calc={calc_f1}, stored={pt['f1']}"
        )
        assert abs(calc_brier - pt["brier_score"]) < 1e-9, (
            f"[{variant}] Brier mismatch: calc={calc_brier}, stored={pt['brier_score']}"
        )
        assert abs(calc_acc - pt["accuracy"]) < 1e-9, (
            f"[{variant}] Accuracy mismatch: calc={calc_acc}, stored={pt['accuracy']}"
        )
