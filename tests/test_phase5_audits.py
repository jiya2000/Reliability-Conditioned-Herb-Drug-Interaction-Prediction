"""
Tests for Phase 5 Debug Pass & Audits.

Verifies:
1. Constant predictor CI is exactly [0.5, 0.5].
2. Every point estimate lies strictly inside its own 95% cluster bootstrap CI.
3. Text masking hierarchy is strictly monotonic:
   full_text >= level1 >= level2 >= level3 >= design_only under pair_disjoint.
4. Nadeau-Bengio resampled paired t-tests and cluster bootstrap difference tests are present.
5. Exact feature breakdowns for LR (8D and 13D) are properly recorded.
"""

import json
import os
import pytest
import numpy as np


@pytest.fixture(scope="module")
def phase5_results():
    path = "results/phase5_task_a.json"
    assert os.path.exists(path), f"{path} does not exist. Run scripts/run_phase5_task_a.py first."
    with open(path, "r") as f:
        return json.load(f)


def test_constant_predictor_ci_is_half(phase5_results):
    """
    Item 2 Requirement:
    Fails if a constant predictor's CI is not [0.5, 0.5].
    """
    bt = phase5_results["baseline_table"]
    for split_mode, models in bt.items():
        maj = models["majority_class"]
        ci_h = maj["ci_95_herb_cluster"]
        ci_d = maj["ci_95_drug_cluster"]
        assert ci_h == [0.5, 0.5], f"majority_class herb CI in {split_mode} was {ci_h}, expected [0.5, 0.5]"
        assert ci_d == [0.5, 0.5], f"majority_class drug CI in {split_mode} was {ci_d}, expected [0.5, 0.5]"


def test_point_estimate_strictly_inside_ci(phase5_results):
    """
    Item 2 & Item 8 Requirement:
    Fails if any point estimate lies outside its own 95% cluster bootstrap CI.
    """
    bt = phase5_results["baseline_table"]
    for split_mode, models in bt.items():
        for m_name, metrics in models.items():
            mean_auc = metrics["mean_auc"]
            ci_h = metrics["ci_95_herb_cluster"]
            ci_d = metrics["ci_95_drug_cluster"]

            # Tolerance of 1e-4 for rounding in JSON
            assert ci_h[0] - 1e-4 <= mean_auc <= ci_h[1] + 1e-4, (
                f"Model {m_name} in {split_mode}: point estimate {mean_auc} outside herb CI {ci_h}"
            )
            assert ci_d[0] - 1e-4 <= mean_auc <= ci_d[1] + 1e-4, (
                f"Model {m_name} in {split_mode}: point estimate {mean_auc} outside drug CI {ci_d}"
            )


def test_text_masking_monotonicity(phase5_results):
    """
    Item 1 & Item 8 Requirement:
    Level 1 >= Level 2 >= Level 3 >= Design-Only under pair_disjoint.
    Level 2 (0.969) > Level 1 (0.946) bug must be resolved.
    """
    bt_pair = phase5_results["baseline_table"]["pair_disjoint"]
    raw_auc = bt_pair["full_text_leaky"]["mean_auc"]
    l1_auc = bt_pair["masked_text_level1"]["mean_auc"]
    l2_auc = bt_pair["masked_text_level2"]["mean_auc"]
    l3_auc = bt_pair["masked_text_level3"]["mean_auc"]
    des_auc = bt_pair["design_only"]["mean_auc"]

    # Raw >= Level 1
    assert raw_auc >= l1_auc - 1e-4, f"Raw {raw_auc} < Level 1 {l1_auc}"
    # Level 1 >= Level 2 (Fix for the Level 2 > Level 1 bug)
    assert l1_auc >= l2_auc - 1e-4, f"Level 1 {l1_auc} < Level 2 {l2_auc} (monotonicity violated)"
    # Level 2 >= Level 3
    assert l2_auc >= l3_auc - 1e-4, f"Level 2 {l2_auc} < Level 3 {l3_auc} (monotonicity violated)"
    # Level 3 >= Design-Only
    assert l3_auc >= des_auc - 1e-4, f"Level 3 {l3_auc} < Design {des_auc}"


def test_nadeau_bengio_and_bootstrap_paired_tests(phase5_results):
    """
    Item 3 Requirement:
    Nadeau-Bengio corrected resampled t-test and bootstrap difference tests present with Holm correction.
    """
    paired = phase5_results.get("paired_tests_vs_lr", {})
    assert "pair_disjoint" in paired
    pair_tests = paired["pair_disjoint"]

    for m_name in ["overlap_6d", "curation_2d", "quality_only", "majority_class"]:
        assert m_name in pair_tests
        t_info = pair_tests[m_name]
        assert "t_stat_nb" in t_info
        assert "p_val_raw_nb" in t_info
        assert "p_val_holm_nb" in t_info
        assert "herb_boot_ci" in t_info
        assert "drug_boot_ci" in t_info
        assert "p_val_holm_herb_boot" in t_info
        assert "p_val_holm_drug_boot" in t_info


def test_lr_feature_breakdown_documented(phase5_results):
    """
    Item 5 Requirement:
    Document LR's exact feature list and verify rows for overlap_6d + curation_2d and overlap_6d + quality.
    """
    meta = phase5_results["metadata"]
    assert "lr_feature_breakdown" in meta
    fb = meta["lr_feature_breakdown"]
    assert len(fb["lr_overlap_plus_curation_8d_names"]) == 8
    assert len(fb["lr_overlap_plus_quality_13d_names"]) == 13

    bt_pair = phase5_results["baseline_table"]["pair_disjoint"]
    assert "lr_overlap_plus_curation" in bt_pair
    assert "lr_overlap_plus_quality" in bt_pair
