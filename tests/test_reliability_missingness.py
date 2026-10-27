"""
Phase 2 Unit Tests: Missing-Literature Reliability Collapse and Evidence Quality Ordering.
Verifies that:
1. When metadata_missingness = 1.0 (no literature evidence exists for a pair), R collapses to ~0.0 (R <= 0.05)
   and z_fused strictly matches z_graph within float tolerance.
2. High-quality human clinical trial evidence generates higher reliability R than in vitro-only evidence.
"""

import numpy as np
import pytest
import torch

from src.models.interaction_model import ReliabilityConditionedInteractionModel
from src.models.reliability_scorer import ReliabilityScorer


def test_missing_literature_collapses_to_graph_prior():
    """
    Theoretical Requirement:
    z_fused = (1 - R) * z_graph + R * z_contextual
    When missingness = 1.0 (de novo, completely unstudied pair with zero literature):
    R must collapse to <= 0.05 (ideally 0.0), so z_fused == z_graph within tolerance.
    """
    graph_dim = 16
    evidence_dim = 16
    fused_dim = 16

    herb_emb = torch.randn(1, graph_dim)
    drug_emb = torch.randn(1, graph_dim)
    # Even if spurious random evidence embeddings exist:
    evidence_feat = torch.randn(1, evidence_dim)

    # 12D metadata with missingness = 1.0 (index 7)
    meta_missing = torch.zeros(1, 12, dtype=torch.float32)
    meta_missing[0, 7] = 1.0  # missingness = 1.0

    model = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim,
        evidence_dim=evidence_dim,
        fused_dim=fused_dim,
        hidden_dim=16,
        mode="reliability_conditioned_quality_only",
    )
    model.eval()

    with torch.no_grad():
        out = model(herb_emb, drug_emb, evidence_feat, metadata=meta_missing, return_details=True)

    r_val = out["reliability"].item()
    z_fused = out["z_fused"]
    z_graph = out["z_graph"]

    # 1. R must be <= 0.05
    assert r_val <= 0.05, f"Expected R <= 0.05 when missingness=1.0, got R={r_val:.4f}"

    # 2. z_fused must match z_graph within numerical tolerance (1e-4)
    assert torch.allclose(z_fused, z_graph, atol=1e-4), (
        f"z_fused failed to collapse to z_graph! Max abs diff: {torch.max(torch.abs(z_fused - z_graph)).item():.6f}"
    )


def test_quality_ordering_human_trial_higher_than_in_vitro():
    """
    Ordering Requirement:
    High-quality human clinical trial evidence must achieve strictly higher reliability R
    than in vitro cell-line or binding assay evidence.
    """
    scorer = ReliabilityScorer(hidden_dim=16, feature_group="quality_only")
    scorer.eval()

    # High-quality human RCT evidence:
    # echo=0.0 (independent), source_quality=1.0 (clinical trial), context_compat=1.0 (human oral),
    # contradiction=0.0, missingness=0.0, rho_reach=0.8, f_m_CYP=0.7
    meta_human_rct = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.8, 0.7]], dtype=torch.float32)

    # In vitro-only evidence:
    # echo=0.3, source_quality=0.30 (cell line), context_compat=0.30 (in vitro assay),
    # contradiction=0.0, missingness=0.0, rho_reach=0.3, f_m_CYP=0.5
    meta_in_vitro = torch.tensor([[0.3, 0.30, 0.30, 0.0, 0.0, 0.3, 0.5]], dtype=torch.float32)

    with torch.no_grad():
        r_human, _ = scorer(meta_human_rct)
        r_invitro, _ = scorer(meta_in_vitro)

    assert r_human.item() > r_invitro.item(), (
        f"Expected R(Human RCT) > R(In Vitro), but got R(Human)={r_human.item():.4f} vs R(In Vitro)={r_invitro.item():.4f}"
    )
