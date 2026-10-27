"""
Unit tests for the Weak Spots and Statistical Hardening Audit (Task 7).

Verifies that:
1. reports/audit_weak_spots.json, .md, and figures/abstention_risk_coverage.png are generated.
2. Component 1: Reliability ordering property holds (R_human > R_invitro) on held-out test records.
3. Component 2: Decision stability across seeds and ensembling stabilization mechanism.
4. Component 3: Risk-coverage curve, AURC, and excess AURC mathematical bounds.
5. Component 4: Pairwise significance tests and Holm-Bonferroni FWER control.
6. Component 5: Repeated grouped 5-fold cross-validation without study leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def audit_data():
    """Loads reports/audit_weak_spots.json."""
    json_path = PROJECT_ROOT / "reports" / "audit_weak_spots.json"
    if not json_path.exists():
        # Run audit if missing
        from benchmarks.audit_weak_spots import run_weak_spots_audit
        run_weak_spots_audit()

    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def test_audit_artifacts_exist():
    """Verify all deliverables of the weak spots audit are persisted."""
    json_path = PROJECT_ROOT / "reports" / "audit_weak_spots.json"
    md_path = PROJECT_ROOT / "reports" / "audit_weak_spots.md"
    fig_path = PROJECT_ROOT / "reports" / "figures" / "abstention_risk_coverage.png"

    assert json_path.exists(), f"Missing {json_path}"
    assert md_path.exists(), f"Missing {md_path}"
    assert fig_path.exists(), f"Missing {fig_path}"

    assert json_path.stat().st_size > 1000
    assert md_path.stat().st_size > 1000
    assert fig_path.stat().st_size > 5000


def test_component1_reliability_ordering(audit_data):
    """Verify Component 1: Reliability ordering property on held-out test splits."""
    c1 = audit_data["component_1_reliability_ordering"]

    # Verify with ordering loss
    ord_res = c1["with_ordering_loss"]
    assert ord_res["ordering_invariance_fraction"] >= 0.95
    assert ord_res["human_mean_R"] > ord_res["invitro_mean_R"]
    assert ord_res["delta_R"] > 0.20

    # Verify without ordering loss
    no_ord_res = c1["without_ordering_loss"]
    assert no_ord_res["ordering_invariance_fraction"] >= 0.95
    assert no_ord_res["human_mean_R"] > no_ord_res["invitro_mean_R"]

    # Verify correlations are finite
    corr = c1["correlation_analysis"]
    assert -1.0 <= corr["pearson_r_correctness"] <= 1.0
    assert -1.0 <= corr["spearman_rho_correctness"] <= 1.0
    assert -1.0 <= corr["pearson_r_brier_loss"] <= 1.0

    # Verify rationale is present and non-empty
    assert len(c1["human_rct_r_rationale"]) > 50
    assert "convex combination" in c1["human_rct_r_rationale"].lower() or "gating" in c1["human_rct_r_rationale"].lower()


def test_component2_decision_stability_and_ensembling(audit_data):
    """Verify Component 2: Decision stability across seeds and ensemble stabilization."""
    c2 = audit_data["component_2_decision_stability_and_ensembling"]

    assert len(c2["seeds_evaluated"]) == 5
    single_stab = c2["single_run_stability"]
    ens_stab = c2["ensemble_stabilization"]

    # Invariance fraction must be valid probability
    assert 0.0 <= single_stab["invariance_fraction"] <= 1.0
    assert 0.0 <= single_stab["pairwise_agreement"] <= 1.0

    # Ensemble must stabilize decisions (higher or equal invariance)
    assert ens_stab["invariance_fraction_leave_one_out"] >= single_stab["invariance_fraction"]
    assert ens_stab["pairwise_agreement_leave_one_out"] >= single_stab["pairwise_agreement"]

    # Flipping pairs must be documented
    flip_pairs = c2["flipping_pairs_detail"]
    assert len(flip_pairs) == single_stab["flipping_pair_count"]
    for fp in flip_pairs:
        assert "herb_id" in fp
        assert "drug_id" in fp
        assert "root_cause" in fp
        assert len(fp["root_cause"]) > 0


def test_component3_risk_coverage_aurc(audit_data):
    """Verify Component 3: Risk-coverage continuous curve and AURC properties."""
    c3 = audit_data["component_3_conformal_risk_coverage"]
    aurc_summary = c3["continuous_aurc_summary"]

    aurc = aurc_summary["aurc"]
    opt_aurc = aurc_summary["optimal_aurc"]
    excess = aurc_summary["excess_aurc"]

    assert 0.0 <= opt_aurc <= aurc <= 1.0
    assert abs((aurc - opt_aurc) - excess) < 1e-3

    # Check parametric sweeps
    sweeps = c3["parametric_grid_sweeps"]
    assert len(sweeps) >= 15  # at least 5 epsilons x 3 conflict thresholds
    for s in sweeps:
        assert 0.0 <= s["coverage"] <= 1.0
        assert 0.0 <= s["retained_accuracy"] <= 1.0
        assert s["n_retained"] + s["n_abstained"] == s["n_total"]


def test_component4_statistical_significance_tests(audit_data):
    """Verify Component 4: Pairwise significance tests and Holm-Bonferroni correction."""
    c4 = audit_data["component_4_statistical_significance_tests"]
    assert c4["num_baselines_compared"] == 8
    assert c4["test_sample_size"] == 26

    comparisons = c4["pairwise_comparisons"]
    assert len(comparisons) == 8

    for comp in comparisons:
        assert "baseline_model" in comp
        assert "delta_auc_prop_minus_base" in comp
        assert 0.0 <= comp["p_value_bootstrap_auc"] <= 1.0
        assert 0.0 <= comp["p_value_delong_auc"] <= 1.0
        assert 0.0 <= comp["p_value_mcnemar"] <= 1.0
        assert 0.0 <= comp["p_value_bootstrap_brier"] <= 1.0
        assert 0.0 <= comp["p_value_holm_bonferroni"] <= 1.0

        # Holm-Bonferroni adjusted p-value must be >= raw p-value
        assert comp["p_value_holm_bonferroni"] >= comp["p_value_bootstrap_auc"] - 1e-6


def test_component5_repeated_grouped_cross_validation(audit_data):
    """Verify Component 5: Repeated grouped 5-fold cross-validation results."""
    c5 = audit_data["component_5_repeated_grouped_cross_validation"]
    assert c5["num_repeats"] == 5
    assert c5["folds_per_repeat"] == 5

    models = c5["models_evaluated"]
    assert len(models) == 5

    model_names = [m["model_name"] for m in models]
    assert "Proposed (Quality-Only)" in model_names
    assert "Naive Concat MLP" in model_names
    assert "DeepDDI" in model_names
    assert "GBT" in model_names
    assert "Logistic Regression" in model_names

    for m in models:
        assert m["folds_evaluated"] >= 20  # almost all 25 folds evaluated
        assert 0.50 <= m["roc_auc_mean"] <= 1.0
        assert 0.0 <= m["roc_auc_std"] <= 0.30
        assert 0.50 <= m["pr_auc_mean"] <= 1.0
        assert 0.0 <= m["brier_mean"] <= 0.50
