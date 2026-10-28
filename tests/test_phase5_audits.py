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

    for m_name in ["overlap_6d", "curation_2d", "lineage_features_7d", "majority_class"]:
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
    Item 6 Requirement:
    Document LR's exact feature list and verify rows for overlap_6d + curation_2d and overlap_6d + sanitized_design.
    """
    meta = phase5_results["metadata"]
    assert "lr_feature_breakdown" in meta
    fb = meta["lr_feature_breakdown"]
    assert len(fb["lr_overlap_plus_curation_8d_names"]) == 8
    assert len(fb["lr_overlap_plus_sanitized_design_12d_names"]) == 12
    assert len(fb["logistic_regression_13d_names"]) == 13

    bt_pair = phase5_results["baseline_table"]["pair_disjoint"]
    assert "lr_overlap_plus_curation" in bt_pair
    assert "lr_overlap_plus_sanitized_design" in bt_pair
    assert "logistic_regression" in bt_pair


def test_constant_predictor_ece_below_threshold(phase5_results):
    """
    Item 2 & 11 Requirement:
    Constant predictor at base rate must have ECE < 0.03 across all split modes.
    """
    bt = phase5_results["baseline_table"]
    for split_mode, models in bt.items():
        maj = models["majority_class"]
        ece = maj["ece_equal_mass"]
        assert ece < 0.03, f"majority_class ECE in {split_mode} was {ece}, expected < 0.03"


def test_no_temperature_at_bound(phase5_results):
    """
    Item 2 & 11 Requirement:
    No model should have a temperature at a bound without being marked 'not identifiable'.
    """
    bt = phase5_results["baseline_table"]
    for split_mode, models in bt.items():
        for m_name, metrics in models.items():
            temp = metrics["temperature"]
            if temp == "not identifiable":
                assert metrics.get("temperature_identifiable") is False
            else:
                assert isinstance(temp, (int, float))
                # Bounds are (0.001, 1000.0); optimum must be interior (not within 5% of bound)
                assert temp > 0.001 * 1.05, f"Model {m_name} in {split_mode} hit lower temperature bound: {temp}"
                assert temp < 1000.0 * 0.95, f"Model {m_name} in {split_mode} hit upper temperature bound: {temp}"


def test_tri_state_abstention_signal_distinction(phase5_results):
    """
    Item 3 & 11 Requirement:
    Tri-state engine must not report an identical row to max-probability without a separate signal;
    under legacy labels where conflict C_{h,d} = 0, the tri-state row is dropped from the matched comparison.
    """
    abst = phase5_results.get("abstention_comparison", {})
    assert "pair_disjoint" in abst
    ac = abst["pair_disjoint"]
    assert ac.get("tri_state_row_dropped") is True
    assert "tri_state_legacy_status" in ac


def test_determinism_run_twice_compare_hashes():
    """
    Item 10 Requirement:
    Determinism test: run twice, compare hashes, without updating expected hash.
    Verifies that running grouped CV splits and model training twice produces
    100% bitwise identical predictions.
    """
    import pandas as pd
    import hashlib
    from sklearn.linear_model import LogisticRegression
    from src.training.evaluation import LeakageSafeSplitter
    from scripts.run_phase5_task_a import extract_all_feature_representations

    df = pd.read_csv("data/benchmark/verified_botanical_hdi.csv")
    feats = extract_all_feature_representations(df)
    y = feats["y"]
    X = feats["X_overlap_6d"]

    hashes = []
    for run in range(2):
        preds = []
        for rep in range(2):
            splitter = LeakageSafeSplitter(seed=42 + rep)
            folds = splitter.grouped_kfold_split(df, mode="pair_disjoint", n_splits=5)
            for tr_df, te_df in folds:
                tr_idx = tr_df.index.values
                te_idx = te_df.index.values
                clf = LogisticRegression(C=1.0, max_iter=200, random_state=42 + rep)
                clf.fit(X[tr_idx], y[tr_idx])
                p_te = clf.predict_proba(X[te_idx])[:, 1]
                preds.extend(p_te.tolist())
        h = hashlib.sha256(json.dumps(preds).encode("utf-8")).hexdigest()
        hashes.append(h)

    assert hashes[0] == hashes[1], f"Non-deterministic execution: hash 1 {hashes[0]} != hash 2 {hashes[1]}"



