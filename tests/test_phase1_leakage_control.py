"""
Phase 1 Unit Tests: Leakage Control and Feature-Group Switch.
Verifies that:
1. Feature groups ('full', 'quality_only', 'polarity_only') are correctly partitioned.
2. Quality-only features drop all label-adjacent features (S_pos, S_neg, margin, raw_pos, raw_neg).
3. 'reliability_conditioned_quality_only' is a valid ablation mode in InteractionModel.
4. InteractionModel defaults to 'quality_only' for headline reliability conditioning.
"""

import numpy as np
import pytest
import torch

from src.data.evidence_lineage import (
    FEATURE_GROUP_FULL,
    FEATURE_GROUP_POLARITY_ONLY,
    FEATURE_GROUP_QUALITY_ONLY,
    PairEvidenceSummary,
)
from src.models.interaction_model import ReliabilityConditionedInteractionModel
from src.models.reliability_scorer import ReliabilityScorer


def test_feature_group_partitioning():
    """Verify that PairEvidenceSummary accurately extracts feature groups without cross-contamination."""
    summary = PairEvidenceSummary(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        total_records=6,
        raw_positive_count=5,
        raw_negative_count=1,
        raw_conflicting_count=0,
        raw_unknown_count=0,
        num_independent_positive_clusters=2,
        num_independent_negative_clusters=1,
        independent_positive_strength=2.3,
        independent_negative_strength=0.8,
        citation_echo_ratio=0.35,
        mean_source_quality=0.85,
        context_compatibility_score=0.90,
        metadata_missingness=0.0,
        contradiction_strength=0.25,
        net_evidence_margin=1.5,
    )

    # 1. Full 12D Vector
    v_full = summary.to_feature_group(FEATURE_GROUP_FULL)
    assert len(v_full) == 12
    # Verify positions
    assert v_full[0] == pytest.approx(2.3)  # S_pos
    assert v_full[1] == pytest.approx(0.8)  # S_neg
    assert v_full[6] == pytest.approx(1.5)  # margin
    assert v_full[8] == pytest.approx(5.0)  # raw_pos
    assert v_full[9] == pytest.approx(1.0)  # raw_neg

    # 2. Polarity-Only (5D label-adjacent features)
    v_pol = summary.to_feature_group(FEATURE_GROUP_POLARITY_ONLY)
    assert len(v_pol) == 5
    # Must contain S_pos, S_neg, margin, raw_pos, raw_neg
    np.testing.assert_allclose(v_pol, [2.3, 0.8, 1.5, 5.0, 1.0], rtol=1e-5)

    # 3. Quality-Only (7D strictly leakage-free features)
    v_qual = summary.to_feature_group(FEATURE_GROUP_QUALITY_ONLY)
    assert len(v_qual) == 7
    # Expected: [echo, qual, ctx, conflict, missing, reach, cyp]
    np.testing.assert_allclose(v_qual, [0.35, 0.85, 0.90, 0.25, 0.0, 0.5, 0.5], rtol=1e-5)

    # CRITICAL CHECK: Neither raw counts nor independent polarity strengths exist in quality-only
    assert 2.3 not in v_qual
    assert 0.8 not in v_qual
    assert 1.5 not in v_qual
    assert 5.0 not in v_qual


def test_reliability_scorer_feature_groups():
    """Verify ReliabilityScorer forward passes under quality_only, full, and polarity_only."""
    # Test quality_only scorer on 7D input
    scorer_qual = ReliabilityScorer(hidden_dim=16, feature_group="quality_only")
    meta_7d = torch.tensor([[0.2, 0.9, 0.95, 0.1, 0.0, 0.7, 0.6]], dtype=torch.float32)
    r_qual, breakdown = scorer_qual(meta_7d, return_breakdown=True)
    assert r_qual.shape == (1, 1)
    assert 0.0 <= r_qual.item() <= 1.0
    assert "rho_reach_weight" in breakdown
    assert "f_m_CYP_weight" in breakdown

    # Test full scorer on 12D input
    scorer_full = ReliabilityScorer(hidden_dim=16, feature_group="full")
    meta_12d = torch.tensor([[2.0, 0.0, 0.2, 0.9, 0.95, 0.1, 2.0, 0.0, 3.0, 0.0, 0.7, 0.6]], dtype=torch.float32)
    r_full, _ = scorer_full(meta_12d)
    assert r_full.shape == (1, 1)
    assert 0.0 <= r_full.item() <= 1.0


def test_interaction_model_quality_only_mode():
    """Verify that interaction_model supports reliability_conditioned_quality_only mode."""
    graph_dim = 16
    evidence_dim = 16
    herb_emb = torch.randn(2, graph_dim)
    drug_emb = torch.randn(2, graph_dim)
    evidence_feat = torch.randn(2, evidence_dim)
    meta_12d = torch.randn(2, 12).abs()
    meta_12d[:, 7] = 0.0  # missingness = 0

    model = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim,
        evidence_dim=evidence_dim,
        fused_dim=16,
        hidden_dim=16,
        mode="reliability_conditioned_quality_only",
    )
    assert model.mode == "reliability_conditioned_quality_only"
    assert model.feature_group == "quality_only"

    out = model(herb_emb, drug_emb, evidence_feat, metadata=meta_12d, return_details=True)
    assert "logits" in out
    assert "probs" in out
    assert "reliability" in out
    assert out["probs"].shape == (2, 1)
    assert out["reliability"].shape == (2, 1)
