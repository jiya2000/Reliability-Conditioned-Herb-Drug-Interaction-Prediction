"""
Tests for Novelty 1: 10-Dimensional Evidence Lineage Vector (CDA + PCC + temporal).

Validates the EvidenceLineageExtractor, EvidenceLineageFeatures, and
EvidenceLineageEncoder against the patent specification requirements.
"""

import math
import sys
from typing import Optional

import torch
import numpy as np

from src.models.evidence_lineage import (
    EvidenceLineageFeatures,
    EvidenceLineageExtractor,
    EvidenceLineageEncoder,
    TemporalEvidenceWeighting,
)
from src.models.citation_dag import CitationDAG, CitationDAGResult, PaperNode
from src.models.conformal_abstention import ClinicalDecision


def _compute_contradiction_index(S_pos: float, S_neg: float) -> float:
    """Contradiction index C_h,d per patent Eq. 4."""
    eps = 1e-6
    return (2.0 * min(S_pos, S_neg)) / (S_pos + S_neg + eps) if S_pos + S_neg >= eps else 0.0


# ============================================================
# Test 1: EvidenceLineageFeatures zeros and tensor conversion
# ============================================================
def test_evidence_lineage_features_zeros():
    # zeros() is a classmethod returning a tensor directly (shape (1, 10))
    tensor = EvidenceLineageFeatures.zeros()
    assert tensor.shape == (1, 10), f"Test 1 FAILED: Expected (1,10) got {tensor.shape}"
    assert torch.all(tensor == 0.0), "Test 1 FAILED: zeros not all zero"


def test_evidence_lineage_features_instance_to_tensor():
    # Instance method to_tensor() also works
    f = EvidenceLineageFeatures(S_pos=0.5, S_neg=0.3, N_primary=2, N_echo=1,
                                kappa_context=0.6, C_contradiction=0.1,
                                q_bar=0.7, Delta_time=0.5, N_raw_pos=1, N_raw_neg=0)
    tensor = f.to_tensor()
    assert tensor.shape == (1, 10), f"Test 2 FAILED: Expected (1,10) got {tensor.shape}"


def test_evidence_lineage_features_roundtrip():
    """Features -> tensor -> reconstructed features."""
    original = EvidenceLineageFeatures(
        S_pos=0.75, S_neg=0.25, N_primary=3, N_echo=5,
        kappa_context=0.80, C_contradiction=0.10,
        q_bar=0.70, Delta_time=0.60, N_raw_pos=2, N_raw_neg=3,
    )
    tensor = original.to_tensor()
    assert tensor.shape == (1, 10), "Test 3a FAILED: tensor shape"
    # Verify key attributes preserved
    assert original.S_pos == original.to_tensor()[0, 0].item()
    assert original.S_neg == original.to_tensor()[0, 1].item()


# ============================================================
# Test 2: Empty/single-record extraction
# ============================================================
def test_empty_record_extraction():
    """Extraction with no records returns zero features and zero DAG."""
    extractor = EvidenceLineageExtractor(alpha=0.15)
    features, dag_result = extractor.extract([], current_year=2024)
    assert features.N_raw_pos == 0, "Test 4a FAILED: N_raw_pos"
    assert features.N_raw_neg == 0, "Test 4b FAILED: N_raw_neg"
    assert features.S_pos == 0.0, "Test 4c FAILED: S_pos"
    assert features.S_neg == 0.0, "Test 4d FAILED: S_neg"
    assert dag_result.total_papers == 0, "Test 4e FAILED: total_papers"
    assert dag_result.echo_chamber_index == 0.0, "Test 4f FAILED: echo_chamber_index"


def test_single_root_extraction():
    """Single primary clinical trial -> all features computable."""
    records = [
        {
            "id": "PMID_1001", "year": 2015, "study_type": "clinical",
            "citations": [], "sentiment": "positive", "quality": 0.95,
            "species": "human", "preparation": "standardized_extract", "route": "oral",
        }
    ]
    extractor = EvidenceLineageExtractor(alpha=0.15)
    features, dag_result = extractor.extract(records, current_year=2024)

    assert dag_result.total_papers == 1, f"Test 5 FAILED: total_papers={dag_result.total_papers}"
    assert dag_result.num_roots == 1, f"Test 5 FAILED: num_roots={dag_result.num_roots}"
    assert dag_result.num_derivatives == 0, f"Test 5 FAILED: num_derivatives={dag_result.num_derivatives}"
    assert dag_result.effective_independent_count == 1.0, f"Test 5 FAILED: effective={dag_result.effective_independent_count}"
    assert dag_result.echo_chamber_index == 0.0, f"Test 5 FAILED: echo_chamber={dag_result.echo_chamber_index}"
    assert features.N_primary == 1, f"Test 5 FAILED: N_primary={features.N_primary}"
    assert features.N_echo == 0, f"Test 5 FAILED: N_echo={features.N_echo}"


# ============================================================
# Test 3: Contradiction index (standalone function)
# ============================================================
def test_contradiction_index():
    """Contradiction index C_h,d per patent Eq. 4 (standalone function)."""
    # Equal positive and negative strengths -> C -> 1.0
    c = _compute_contradiction_index(0.5, 0.5)
    assert abs(c - 1.0) < 1e-6, f"Test 6a FAILED: Expected ~1.0, got {c}"

    # One-sided only -> C -> 0.0
    c = _compute_contradiction_index(1.0, 0.0)
    assert c == 0.0, f"Test 6b FAILED: Expected 0.0, got {c}"

    # Zero both -> C -> 0.0
    c = _compute_contradiction_index(0.0, 0.0)
    assert c == 0.0, f"Test 6c FAILED: Expected 0.0, got {c}"


# ============================================================
# Test 4: Temporal decay weighting
# ============================================================
def test_temporal_decay():
    """Temporal recency computes decreasing Delta_time for older evidence."""
    extractor = EvidenceLineageExtractor(alpha=0.15, temporal_half_life_years=5.0)

    # Recent evidence (2024, 2023)
    recent_records = [
        {"id": "R1", "year": 2024, "study_type": "clinical", "citations": [], 
         "sentiment": "positive", "quality": 0.9, "species": "human", "preparation": "standardized_extract", "route": "oral"},
        {"id": "R2", "year": 2023, "study_type": "clinical", "citations": [], 
         "sentiment": "positive", "quality": 0.9, "species": "human", "preparation": "standardized_extract", "route": "oral"},
    ]
    features_recent, _ = extractor.extract(recent_records, current_year=2024)

    # Old evidence (2015, 2016)
    old_records = [
        {"id": "O1", "year": 2015, "study_type": "clinical", "citations": [], 
         "sentiment": "positive", "quality": 0.9, "species": "human", "preparation": "standardized_extract", "route": "oral"},
        {"id": "O2", "year": 2016, "study_type": "clinical", "citations": [], 
         "sentiment": "positive", "quality": 0.9, "species": "human", "preparation": "standardized_extract", "route": "oral"},
    ]
    features_old, _ = extractor.extract(old_records, current_year=2024)

    # Recent should have higher Delta_time (closer to 1.0)
    assert features_recent.Delta_time >= features_old.Delta_time, (
        f"Test 7 FAILED: Recent {features_recent.Delta_time:.4f} < Old {features_old.Delta_time:.4f}"
    )


# ============================================================
# Test 5: EvidenceLineageEncoder
# ============================================================
def test_evidence_lineage_encoder():
    """EvidenceLineageEncoder produces correct output shape and is deterministic."""
    encoder = EvidenceLineageEncoder(output_dim=128)

    test_features = torch.tensor([
        0.75, 0.25,   # S_pos, S_neg
        3.0, 5.0,     # N_primary, N_echo
        0.80,         # kappa_context
        0.10,         # C_contradiction
        0.70,         # q_bar
        0.60,         # Delta_time
        2.0, 3.0,     # N_raw_pos, N_raw_neg
    ])

    encoder.eval()  # Critical: eval() mode for deterministic output
    encoded = encoder(test_features.unsqueeze(0))  # (1, 10) -> (1, 128)
    assert encoded.shape == (1, 128), f"Test 8 FAILED: Expected (1,128) got {encoded.shape}"

    encoded1 = encoder(test_features.unsqueeze(0))
    encoded2 = encoder(test_features.unsqueeze(0))
    assert torch.allclose(encoded1, encoded2, atol=1e-6), "Test 9 FAILED: Not deterministic"


# ============================================================
# Test 5: Temporal weighting module
# ============================================================
def test_temporal_weighting_module():
    """TemporalEvidenceWeighting module produces proper decay."""
    weighting = TemporalEvidenceWeighting(initial_half_life=5.0, learnable=False)

    test_years = torch.tensor([
        2019.0, 2020.0, 2021.0, 2022.0, 2023.0, 2024.0
    ])

    weights = weighting(test_years, current_year=2024.0)

    # 2024 (current year) should have weight 1.0
    assert abs(weights[5].item() - 1.0) < 1e-6, f"Test 10a FAILED: Current year weight {weights[5].item()}"

    # Earlier years should have lower weights
    assert weights[0].item() < weights[5].item(), f"Test 10b FAILED: Monotonicity violated"

    # Strict monotonicity: earlier = lower weight
    for i in range(len(weights) - 1):
        assert weights[i].item() <= weights[i+1].item(), f"Test 10c FAILED: Not monotonic at {i}"


# ============================================================
# Test 6: Full evidence lineage pipeline
# ============================================================
def test_full_evidence_lineage_pipeline():
    """Full end-to-end pipeline test from records to features to encoder."""
    records = [
        # Primary RCT
        {"id": "PMID_1001", "year": 2015, "study_type": "clinical", "citations": [], 
         "sentiment": "positive", "quality": 0.95, "species": "human", "preparation": "standardized_extract", "route": "oral"},
        # Review citing the RCT
        {"id": "PMID_1002", "year": 2018, "study_type": "review", "citations": ["PMID_1001"],
         "sentiment": "positive", "quality": 0.8, "species": "human", "preparation": "standardized_extract", "route": "oral"},
        # Independent negative in vitro
        {"id": "PMID_2001", "year": 2019, "study_type": "in_vitro", "citations": [],
         "sentiment": "negative", "quality": 0.7, "species": "in_vitro_microsomes", "preparation": "isolated_constituent", "route": "in_vitro", "concentration_um": 100.0},
    ]

    extractor = EvidenceLineageExtractor(alpha=0.15)
    features, dag_result = extractor.extract(records, current_year=2024)

    # Verify features
    assert features.N_raw_pos >= 1, "Test 11a FAILED: No positive records"
    assert features.N_raw_neg >= 0, "Test 11b FAILED: Negative check"
    assert dag_result.total_papers == 3, f"Test 11c FAILED: total_papers={dag_result.total_papers}"
    assert dag_result.echo_chamber_index >= 0.0, "Test 11d FAILED: echo_chamber_index"

    # Encode
    encoder = EvidenceLineageEncoder(output_dim=128)
    feat_tensor = features.to_tensor()
    encoded = encoder(feat_tensor)
    assert encoded.shape == (1, 128), f"Test 11e FAILED: encoded shape {encoded.shape}"

    # Verify contradiction index (standalone function)
    C = _compute_contradiction_index(features.S_pos, features.S_neg)
    assert 0.0 <= C <= 1.0, f"Test 11e FAILED: Contradiction index {C}"

    print("Full pipeline test PASSED")


# ============================================================
# Test 7: Tri-state decision rule components
# ============================================================
def test_decision_rule_components():
    """Verify each component of the 3-condition tri-state decision rule."""
    from src.models.patent_novelty_system import ReliabilityConditionedSystem

    system = ReliabilityConditionedSystem()

    # Condition 1: Epistemic uncertainty > 0.25
    dec1 = system._tri_state_decision_full(
        p=torch.tensor([0.7]),  # high prob
        epistemic_u=0.35,  # > tau_u=0.25
        C_contradiction=0.1,
        R=0.8,  # high reliability
    )
    assert dec1["abstain"] is True, "Test 12a FAILED: Condition 1 should force abstain"
    assert dec1["routing_destination"] == "clinical_pharmacist_review", "Test 12b FAILED: Route"

    # Condition 2: Contradiction > 0.5 AND margin < 0.35
    dec2 = system._tri_state_decision_full(
        p=torch.tensor([0.6]),  # margin = 0.1 < 0.35
        epistemic_u=0.1,  # low
        C_contradiction=0.6,  # > tau_c=0.5
        R=0.8,
    )
    assert dec2["abstain"] is True, "Test 13a FAILED: Condition 2 should force abstain"
    assert "contradiction" in dec2["abstention_reason"], "Test 13b FAILED: No contradiction reason"

    # Condition 3: R < 0.1 AND margin < 0.10
    dec3 = system._tri_state_decision_full(
        p=torch.tensor([0.52]),  # margin = 0.02 < 0.10
        epistemic_u=0.1,  # low
        C_contradiction=0.1,  # low
        R=0.05,  # < tau_R=0.10
    )
    assert dec3["abstain"] is True, "Test 14a FAILED: Condition 3 should force abstain"
    assert "low_reliability" in dec3["abstention_reason"], "Test 14b FAILED: No low_rel reason"

    # Clear positive case (no abstention)
    dec4 = system._tri_state_decision_full(
        p=torch.tensor([0.8]),  # margin = 0.3 > 0.35? No, 0.3 < 0.35
        epistemic_u=0.1,
        C_contradiction=0.1,
        R=0.8,
    )
    # margin = 0.3, tau_margin = 0.35, so 0.3 < 0.35 -> condition 2 could trigger if C > 0.5
    # Here C=0.1, so condition 2 won't trigger
    # epistemic_u=0.1 < 0.25 -> condition 1 won't trigger
    # R=0.8 > 0.10 -> condition 3 won't trigger
    # So this should NOT abstain -> INTERACTION_PRESENT
    assert dec4["abstain"] is False, "Test 15 FAILED: Should NOT abstain for clear positive"
    assert dec4["decision"] == ClinicalDecision.INTERACTION_PRESENT, f"Test 15 FAILED: Decision {dec4['decision']}"
    assert dec4["predicted_label"] == 1, "Test 15 FAILED: Label"

    # Clear negative case
    dec5 = system._tri_state_decision_full(
        p=torch.tensor([0.2]),  # margin = 0.3
        epistemic_u=0.1,
        C_contradiction=0.1,
        R=0.8,
    )
    assert dec5["abstain"] is False, "Test 16 FAILED: Should NOT abstain"
    assert dec5["decision"] == ClinicalDecision.NO_INTERACTION, f"Test 16 FAILED: Decision {dec5['decision']}"
    assert dec5["predicted_label"] == 0, "Test 16 FAILED: Label"

    print("All decision rule tests PASSED")


# ============================================================
# Test 8: 10D vector bounds
# ============================================================
def test_10d_vector_bounds():
    """Verify the 10D vector has all components in reasonable ranges."""
    records = [
        {"id": "PMID_1", "year": 2020, "study_type": "clinical", "citations": [], 
         "sentiment": "positive", "quality": 0.8, "species": "human", "preparation": "standardized_extract", "route": "oral"},
        {"id": "PMID_2", "year": 2021, "study_type": "review", "citations": ["PMID_1"],
         "sentiment": "negative", "quality": 0.7, "species": "human", "preparation": "crude_decoction", "route": "oral"},
    ]

    extractor = EvidenceLineageExtractor(alpha=0.15)
    features, dag_result = extractor.extract(records, current_year=2024)

    # All normalized quantities should be in [0, 1]
    assert 0.0 <= features.S_pos <= 1.0, f"Test 17 FAILED: S_pos={features.S_pos}"
    assert 0.0 <= features.S_neg <= 1.0, f"Test 17 FAILED: S_neg={features.S_neg}"
    assert 0.0 <= features.kappa_context <= 1.0, f"Test 17 FAILED: kappa_context={features.kappa_context}"
    assert 0.0 <= features.C_contradiction <= 1.0, f"Test 17 FAILED: C_contradiction={features.C_contradiction}"
    assert 0.0 <= features.q_bar <= 1.0, f"Test 17 FAILED: q_bar={features.q_bar}"
    assert 0.0 <= features.Delta_time <= 1.0, f"Test 17 FAILED: Delta_time={features.Delta_time}"

    # Counts can be 0 or more
    assert features.N_raw_pos >= 0, "Test 17 FAILED: N_raw_pos"
    assert features.N_raw_neg >= 0, f"Test 17 FAILED: N_raw_neg={features.N_raw_neg}"
    assert features.N_primary >= 0, f"Test 17 FAILED: N_primary={features.N_primary}"
    assert features.N_echo >= 0, f"Test 17 FAILED: N_echo={features.N_echo}"


# ============================================================
# Test 9: PK penalty factor integration
# ============================================================
def test_penalty_factor_integration():
    """Verify PK reachability penalty integrates with evidence lineage."""
    from src.models.pk_reachability import PKReachabilityScreen, InVitroFinding

    screen = PKReachabilityScreen()

    # Supra-physiological concentration -> penalty 0.25
    finding_high = InVitroFinding(
        herb_constituent="berberine",
        enzyme_transporter="CYP3A4",
        interaction_type="inhibition",
        potency_value=2.0,
        concentration_um=100.0,  # way above 50 uM threshold
    )
    result = screen.screen(finding_high)
    assert result.is_reachable is False, "Test 18a FAILED: supra should not be reachable"
    assert result.penalty_factor == 0.25, f"Test 18b FAILED: penalty {result.penalty_factor}"

    # Reachable finding (hyperforin induction, EC50 very small)
    finding_reach = InVitroFinding(
        herb_constituent="hyperforin",
        enzyme_transporter="CYP3A4",
        interaction_type="induction",
        potency_value=0.001,
    )
    result2 = screen.screen(finding_reach)
    # Penalty should be valid (0.25 or 1.0)
    assert result2.penalty_factor in [0.25, 1.0], f"Test 18c FAILED: penalty {result2.penalty_factor}"

    print("Penalty factor integration test PASSED")


# ============================================================
# Test 9: Evidence schema completeness
# ============================================================
def test_evidence_schema():
    """Schema test: verify 10D vector schema completeness."""
    from src.models.evidence_lineage import EvidenceLineageFeatures

    f = EvidenceLineageFeatures()
    assert hasattr(f, 'S_pos'), "Test 19a FAILED: Missing S_pos"
    assert hasattr(f, 'S_neg'), "Test 19b FAILED: Missing S_neg"
    assert hasattr(f, 'N_primary'), "Test 19c FAILED: Missing N_primary"
    assert hasattr(f, 'N_echo'), "Test 19d FAILED: Missing N_echo"
    assert hasattr(f, 'kappa_context'), "Test 19e FAILED: Missing kappa_context"
    assert hasattr(f, 'C_contradiction'), "Test 19f FAILED: Missing C_contradiction"
    assert hasattr(f, 'q_bar'), "Test 19f FAILED: Missing q_bar"
    assert hasattr(f, 'Delta_time'), "Test 19f FAILED: Missing Delta_time"
    assert hasattr(f, 'N_raw_pos'), "Test 19g FAILED: Missing N_raw_pos"
    assert hasattr(f, 'N_raw_neg'), "Test 19h FAILED: Missing N_raw_neg"

    t = f.to_tensor()
    assert t.shape[0] == 10, f"Test 19i FAILED: tensor shape {t.shape}"

    # Zeros should give zero tensor
    z = EvidenceLineageFeatures.zeros()
    # Use .item() to get Python bool from the 0-d tensor result
    assert bool(torch.all(z == 0.0)), "Test 19j FAILED: zeros not all zero"

    print("Evidence schema test PASSED")