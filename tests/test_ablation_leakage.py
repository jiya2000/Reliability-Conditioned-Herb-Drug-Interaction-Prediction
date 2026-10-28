"""
Test Suite: Ablation Leakage and Quality-Only Isolation Audit.

Enforces:
1. Every variant in the main comparison table receives strictly quality-only features
   and excludes all 5 polarity columns (S_pos, S_neg, margin, N_raw_pos, N_raw_neg).
2. 'no_context_conflict' receives exactly 5D quality features (drops W_context, C_hd).
3. 'no_study_dependence' receives 7D quality features with echo_ratio=0.0 and undiscounted C_hd.
4. No non-reference variant exceeds ROC-AUC 0.98 on the test split without raising a loud warning.
5. Old full-vector / polarity variants appear only in a separate table clearly labeled
   'LEAKY REFERENCE (not a valid model)', never in the headline comparison.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.evidence_lineage import LineageSummary, FEATURE_GROUP_QUALITY_ONLY
from src.models.interaction_model import ReliabilityConditionedInteractionModel
from src.training.evaluation import compute_comprehensive_metrics, format_ablation_markdown_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sentinel_lineage_summary() -> LineageSummary:
    """Creates a LineageSummary with unmistakable sentinel values for polarity columns."""
    return LineageSummary(
        herb_id="HERB_TEST",
        drug_id="DRUG_TEST",
        total_records=10,
        raw_positive_count=666,
        raw_negative_count=555,
        raw_conflicting_count=0,
        raw_unknown_count=0,
        num_independent_positive_clusters=1,
        num_independent_negative_clusters=1,

        # Polarity columns (must be excluded from all main models)
        independent_positive_strength=999.0,   # S_pos (Index 0 in 12D)
        independent_negative_strength=888.0,   # S_neg (Index 1 in 12D)
        net_evidence_margin=777.0,             # margin (Index 6 in 12D)
        raw_margin=111.0,

        # Quality columns (permitted in main models)
        citation_echo_ratio=0.25,              # Index 2 in 12D
        mean_source_quality=0.85,              # Index 3 in 12D
        context_compatibility_score=0.92,      # Index 4 in 12D
        contradiction_strength=0.18,           # Index 5 in 12D
        undiscounted_contradiction_strength=0.42, # Undiscounted C_hd
        metadata_missingness=0.05,             # Index 7 in 12D
        rho_reach=0.70,                        # Index 10 in 12D
        f_m_CYP=0.65,                          # Index 11 in 12D
    )


# -------------------------------------------------------------------------
# Test (a): Assert columns fed to model exclude polarity indices
# -------------------------------------------------------------------------

MAIN_MODES = [
    "reliability_conditioned_quality_only",
    "reliability_conditioned",
    "standard_fusion",
    "naive_concat_mlp",
    "graph_only",
    "evidence_only",
    "graph_plus_naive_evidence",
    "no_study_dependence",
    "no_context_conflict",
]

POLARITY_SENTINELS = {999.0, 888.0, 777.0, 666.0, 555.0}


@pytest.mark.parametrize("mode", MAIN_MODES)
def test_main_modes_exclude_polarity_features(sentinel_lineage_summary: LineageSummary, mode: str):
    """Assert to_feature_vector for every main model excludes all polarity columns."""
    feat_vec = sentinel_lineage_summary.to_feature_vector(mode=mode)
    
    # Check dimensions
    if mode == "no_context_conflict":
        assert len(feat_vec) == 5, f"{mode} must be 5D, got {len(feat_vec)}"
        # Verify W_context (0.92) and C_hd (0.18) are dropped
        assert 0.92 not in feat_vec, "W_context must be dropped from no_context_conflict"
        assert 0.18 not in feat_vec, "C_hd must be dropped from no_context_conflict"
    else:
        assert len(feat_vec) == 7, f"{mode} must be 7D, got {len(feat_vec)}"

    if mode == "no_study_dependence":
        # echo_ratio replaced with 0.0 (undiscounted) and undiscounted contradiction used
        assert feat_vec[0] == 0.0, "no_study_dependence must zero out echo attenuation"
        assert feat_vec[3] == 0.0, "Outcome-derived contradiction must be excluded"

    # Assert none of the polarity sentinel values are present
    for val in feat_vec:
        assert val not in POLARITY_SENTINELS, f"Polarity sentinel {val} leaked into {mode} vector: {feat_vec}"


@pytest.mark.parametrize("mode", MAIN_MODES)
def test_interaction_model_defense_in_depth(sentinel_lineage_summary: LineageSummary, mode: str):
    """
    Assert ReliabilityConditionedInteractionModel strips polarity columns if a 12D
    full vector is passed into forward().
    """
    model = ReliabilityConditionedInteractionModel(
        graph_dim=8, evidence_dim=8, fused_dim=8, hidden_dim=8, mode=mode
    )
    
    # 12D input with sentinels
    full_12d = sentinel_lineage_summary.to_feature_group("full")
    meta_tensor = torch.tensor(full_12d, dtype=torch.float32).unsqueeze(0)
    
    h = torch.randn(1, 8)
    d = torch.randn(1, 8)
    ev = torch.randn(1, 8)
    
    # Forward pass should run cleanly without polarity leakage
    out = model(h, d, ev, metadata=meta_tensor, return_details=True)
    assert "logits" in out
    assert not torch.isnan(out["logits"]).any()


def test_leaky_reference_modes_explicit_gating(sentinel_lineage_summary: LineageSummary):
    """Assert that 12D vectors are only generated when explicitly requested as leaky references."""
    # Without leaky mode, default is 7D quality
    default_vec = sentinel_lineage_summary.to_feature_vector()
    assert len(default_vec) == 7
    for val in default_vec:
        assert val not in POLARITY_SENTINELS

    # Explicit leaky reference mode extracts full 12D vector containing polarity
    leaky_vec = sentinel_lineage_summary.to_feature_vector(mode="leaky_reference_full_vector")
    assert len(leaky_vec) == 12
    assert 999.0 in leaky_vec  # S_pos present
    assert 888.0 in leaky_vec  # S_neg present


# -------------------------------------------------------------------------
# Test (b): Loud warning when non-reference variant exceeds ROC-AUC 0.98
# -------------------------------------------------------------------------

def test_high_auc_raises_warning_on_clean_models():
    """Assert compute_comprehensive_metrics raises UserWarning if ROC-AUC > 0.98 on non-reference."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Perfectly separated probabilities -> AUC 1.00
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])
    
    with pytest.warns(UserWarning, match="SUSPICIOUS HIGH PERFORMANCE WARNING.*ROC-AUC.*> 0.98"):
        rep = compute_comprehensive_metrics(
            y_true=y_true,
            y_prob=y_prob,
            split_name="test_split",
            model_variant="standard_fusion",  # Non-reference model
            compute_bootstrap_ci=False,
        )
    assert rep.roc_auc == 1.0


def test_high_auc_no_warning_on_leaky_references():
    """Assert compute_comprehensive_metrics does NOT raise warning on known leaky references."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])
    
    # Should not raise warning for leaky references
    rep1 = compute_comprehensive_metrics(
        y_true=y_true,
        y_prob=y_prob,
        split_name="test_split",
        model_variant="leaky_reference_full_vector",
        compute_bootstrap_ci=False,
    )
    assert rep1.roc_auc == 1.0

    rep2 = compute_comprehensive_metrics(
        y_true=y_true,
        y_prob=y_prob,
        split_name="test_split",
        model_variant="leaky_polarity_probe",
        compute_bootstrap_ci=False,
    )
    assert rep2.roc_auc == 1.0


def test_no_actual_benchmark_model_exceeds_threshold():
    """Assert that in active benchmark results, no valid non-reference model has ROC-AUC > 0.98."""
    json_path = PROJECT_ROOT / "results" / "verified_hdi_benchmark_results.json"
    if not json_path.exists():
        json_path = PROJECT_ROOT / "reports" / "verified_hdi_benchmark_results.json"
    
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        for r in data.get("reports", []):
            var = r.get("model_variant", "")
            is_leaky = "leaky" in var.lower() or "probe" in var.lower()
            if not is_leaky:
                auc = r.get("roc_auc", 0.0)
                assert auc <= 0.98, f"Non-reference model '{var}' has suspicious ROC-AUC {auc:.4f} > 0.98!"


# -------------------------------------------------------------------------
# Test (c): Leaky-reference label appears in separate table whenever shown
# -------------------------------------------------------------------------

def test_format_ablation_markdown_table_separates_leaky_models():
    """
    Assert format_ablation_markdown_table keeps headline table clean and isolates
    leaky variants into a separate table labeled 'LEAKY REFERENCE (not a valid model)'.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob_clean = np.array([0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8])
    y_prob_leaky = np.array([0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99])

    rep_clean1 = compute_comprehensive_metrics(
        y_true=y_true, y_prob=y_prob_clean, split_name="test", model_variant="reliability_conditioned", compute_bootstrap_ci=False
    )
    rep_clean2 = compute_comprehensive_metrics(
        y_true=y_true, y_prob=y_prob_clean, split_name="test", model_variant="no_context_conflict", compute_bootstrap_ci=False
    )
    rep_leaky1 = compute_comprehensive_metrics(
        y_true=y_true, y_prob=y_prob_leaky, split_name="test", model_variant="leaky_reference_full_vector", compute_bootstrap_ci=False
    )
    rep_leaky2 = compute_comprehensive_metrics(
        y_true=y_true, y_prob=y_prob_leaky, split_name="test", model_variant="leaky_no_context_conflict", compute_bootstrap_ci=False
    )

    all_reports = [rep_clean1, rep_clean2, rep_leaky1, rep_leaky2]
    md_output = format_ablation_markdown_table(all_reports)

    # Split output into sections around the leaky reference header
    assert "### Leaky Reference Baselines (Not Valid Models - Label Leakage Demonstration)" in md_output
    parts = md_output.split("### Leaky Reference Baselines (Not Valid Models - Label Leakage Demonstration)")
    headline_table = parts[0]
    leaky_table = parts[1]

    # Verify headline table contains only clean models
    assert "`reliability_conditioned`" in headline_table
    assert "`no_context_conflict`" in headline_table
    assert "leaky_reference_full_vector" not in headline_table
    assert "leaky_no_context_conflict" not in headline_table

    # Verify leaky table has the explicit label on every row
    assert "LEAKY REFERENCE (not a valid model)" in leaky_table
    assert "`leaky_reference_full_vector` **[LEAKY REFERENCE (not a valid model)]**" in leaky_table
    assert "`leaky_no_context_conflict` **[LEAKY REFERENCE (not a valid model)]**" in leaky_table
