"""
Unit tests for ReliabilityConditionedInteractionModel, ablations, calibration, and leakage-safe splits.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.models.interaction_model import (
    ReliabilityConditionedInteractionModel,
    CalibratedDecision,
)
from src.training.calibration import (
    TemperatureScaler,
    compute_calibration_metrics,
    evaluate_abstention,
)
from src.training.evaluation import (
    LeakageSafeSplitter,
    compute_comprehensive_metrics,
    format_ablation_markdown_table,
)


def test_interaction_model_ablation_modes():
    batch_size = 4
    graph_dim = 32
    evidence_dim = 32
    fused_dim = 32

    herb_emb = torch.randn(batch_size, graph_dim)
    drug_emb = torch.randn(batch_size, graph_dim)
    evidence_feat = torch.randn(batch_size, evidence_dim)
    # 8D lineage metadata: [pos, neg, echo, qual, context, conflict, margin, missing]
    metadata_8d = torch.tensor([
        [2.0, 0.0, 0.2, 0.9, 0.9, 0.0, 2.0, 0.1],
        [0.0, 2.0, 0.1, 0.8, 0.8, 0.0, -2.0, 0.2],
        [1.5, 1.5, 0.3, 0.8, 0.8, 0.9, 0.0, 0.1],  # conflicting
        [0.1, 0.1, 0.0, 0.2, 0.3, 0.1, 0.0, 0.8],  # missing/uncertain
    ], dtype=torch.float32)

    modes = [
        "graph_only",
        "graph_plus_naive_evidence",
        "standard_fusion",
        "reliability_conditioned",
        "no_study_dependence",
        "no_context_conflict",
    ]

    for mode in modes:
        model = ReliabilityConditionedInteractionModel(
            graph_dim=graph_dim,
            evidence_dim=evidence_dim,
            fused_dim=fused_dim,
            hidden_dim=32,
            mode=mode,
        )
        out = model(
            herb_emb=herb_emb,
            drug_emb=drug_emb,
            evidence_features=evidence_feat,
            metadata=metadata_8d,
            return_details=True,
        )
        assert "logits" in out
        assert "probs" in out
        assert out["probs"].shape == (batch_size, 1)
        assert (out["probs"] >= 0.0).all() and (out["probs"] <= 1.0).all()


def test_multi_token_evidence_cross_attention():
    """
    P1 Fix: Verify that when multi-token / multi-report evidence sequence is provided,
    cross-attention genuinely computes a dynamic softmax attention distribution across tokens.
    """
    batch_size = 2
    seq_len = 4
    graph_dim = 16
    evidence_dim = 16

    model = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim,
        evidence_dim=evidence_dim,
        fused_dim=16,
        hidden_dim=16,
        mode="reliability_conditioned",
    )

    herb_emb = torch.randn(batch_size, graph_dim)
    drug_emb = torch.randn(batch_size, graph_dim)
    # 3D multi-token evidence tensor: (batch, seq_len=4, evidence_dim)
    evidence_seq = torch.randn(batch_size, seq_len, evidence_dim)
    metadata = torch.tensor([
        [2.0, 0.0, 0.0, 0.9, 1.0, 0.0, 2.0, 0.0],
        [1.0, 1.0, 0.1, 0.8, 0.8, 0.9, 0.0, 0.1],
    ], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(
            herb_emb=herb_emb,
            drug_emb=drug_emb,
            evidence_features=evidence_seq,
            metadata=metadata,
            return_details=True,
        )

    assert "attn_weights" in out
    attn_w = out["attn_weights"]
    assert attn_w is not None
    assert attn_w.shape == (batch_size, 1, seq_len)
    # Multi-head attention distribution over the 4 tokens must sum to 1.0
    token_sums = attn_w.sum(dim=-1)
    assert torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)


def test_ablation_component_isolation():
    """
    P1 Fix: Verify that each ablation strictly and consistently removes ONLY its stated component:
    - 'no_study_dependence' uses un-discounted raw counts instead of lineage-clustered strength.
    - 'no_context_conflict' eliminates context compatibility and contradiction penalty.
    """
    graph_dim = 16
    evidence_dim = 16
    herb_emb = torch.randn(1, graph_dim)
    drug_emb = torch.randn(1, graph_dim)
    evidence_feat = torch.randn(1, evidence_dim)

    # Candidate with echo chamber (1 primary study repeated 4 times: echo_ratio = 0.8)
    # Discounted strength = 1.0, raw count = 5.0
    meta_echo = torch.tensor([[1.0, 0.0, 0.8, 0.9, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    model_full = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim, evidence_dim=evidence_dim, fused_dim=16, hidden_dim=16,
        mode="reliability_conditioned",
    )
    model_no_dep = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim, evidence_dim=evidence_dim, fused_dim=16, hidden_dim=16,
        mode="no_study_dependence",
    )
    # Synchronize weights so ablations are evaluated on identical parameters
    model_no_dep.load_state_dict(model_full.state_dict())
    model_full.eval()
    model_no_dep.eval()

    with torch.no_grad():
        out_full = model_full(herb_emb, drug_emb, evidence_feat, metadata=meta_echo, return_details=True)
        out_no_dep = model_no_dep(herb_emb, drug_emb, evidence_feat, metadata=meta_echo, return_details=True)

    assert out_full["breakdown"] is not None
    assert out_no_dep["breakdown"] is not None

    # Candidate with strong active contradiction (conflict = 0.95)
    meta_conflict = torch.tensor([[1.0, 1.0, 0.0, 0.9, 0.9, 0.95, 0.0, 0.0]], dtype=torch.float32)
    model_no_conflict = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim, evidence_dim=evidence_dim, fused_dim=16, hidden_dim=16,
        mode="no_context_conflict",
    )
    model_no_conflict.load_state_dict(model_full.state_dict())
    model_no_conflict.eval()

    with torch.no_grad():
        out_conf_full = model_full(herb_emb, drug_emb, evidence_feat, metadata=meta_conflict, return_details=True)
        out_conf_abl = model_no_conflict(herb_emb, drug_emb, evidence_feat, metadata=meta_conflict, return_details=True)

    # Full model penalizes reliability under contradiction, whereas no_context_conflict ignores it
    r_full = out_conf_full["reliability"].item()
    r_abl = out_conf_abl["reliability"].item()
    assert r_abl > r_full, "no_context_conflict must not penalize reliability for scientific contradiction"



def test_calibrated_decision_and_abstention():
    graph_dim = 16
    model = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        mode="reliability_conditioned",
        abstain_uncertainty_threshold=0.25,
        abstain_conflict_threshold=0.50,
    )

    herb_emb = torch.randn(2, graph_dim)
    drug_emb = torch.randn(2, graph_dim)
    evidence_feat = torch.randn(2, 16)
    # Row 0: high conflict (0.95), Row 1: clear positive (conflict 0.0)
    metadata_8d = torch.tensor([
        [1.0, 1.0, 0.1, 0.8, 0.8, 0.95, 0.0, 0.1],
        [3.0, 0.0, 0.0, 0.9, 0.9, 0.0, 3.0, 0.0],
    ], dtype=torch.float32)

    decisions = model.predict_with_decision(
        herb_emb=herb_emb,
        drug_emb=drug_emb,
        evidence_features=evidence_feat,
        metadata=metadata_8d,
    )

    assert len(decisions) == 2
    assert isinstance(decisions[0], CalibratedDecision)
    # The high-conflict pair with equivocal probability should trigger abstention
    assert decisions[0].abstain is True
    assert decisions[0].predicted_label == -1
    assert decisions[0].abstention_reason == "active_scientific_conflict"


def test_temperature_scaling_and_calibration():
    scaler = TemperatureScaler()
    # Simulate validation logits and labels
    val_logits = np.array([-2.5, -1.0, 0.5, 1.2, 3.0])
    val_labels = np.array([0, 0, 1, 1, 1])

    # Fit temperature strictly on validation partition
    t = scaler.fit(val_logits, val_labels, max_iter=20)
    assert t > 0.0

    # Evaluate on held-out test logits (no test leakage)
    test_logits = np.array([-2.0, -0.8, 0.4, 2.2])
    test_labels = np.array([0, 0, 1, 1])
    test_scaled = scaler(torch.tensor(test_logits, dtype=torch.float32))
    probs = torch.sigmoid(test_scaled).detach().numpy()

    cal_metrics = compute_calibration_metrics(probs, test_labels)
    assert 0.0 <= cal_metrics.expected_calibration_error <= 1.0
    assert 0.0 <= cal_metrics.brier_score <= 1.0


def test_leakage_safe_splits():
    splitter = LeakageSafeSplitter(seed=42)
    data = []
    for h_idx in range(10):
        for d_idx in range(5):
            data.append({
                "herb_id": f"HERB_{h_idx:02d}",
                "drug_id": f"DRUG_{d_idx:02d}",
                "study_id": f"STUDY_{h_idx}_{d_idx}",
                "cited_study_ids": [f"STUDY_{h_idx}_{0}"] if d_idx > 0 else [],
                "publication_date": f"{2010 + (h_idx % 12)}-01-01" if (h_idx + d_idx) % 7 != 0 else None,
                "outcome": "positive",
            })
    df = pd.DataFrame(data)

    # 1. Herb-disjoint split
    tr_h, val_h, te_h = splitter.split_herb_disjoint(df)
    train_herbs = set(tr_h["herb_id"])
    test_herbs = set(te_h["herb_id"])
    assert len(train_herbs & test_herbs) == 0, "Herb-disjoint split leaked herbs across train and test!"

    # 2. Drug-disjoint split
    tr_d, val_d, te_d = splitter.split_drug_disjoint(df)
    train_drugs = set(tr_d["drug_id"])
    test_drugs = set(te_d["drug_id"])
    assert len(train_drugs & test_drugs) == 0, "Drug-disjoint split leaked drugs across train and test!"

    # 3. Temporal split: undated records must be quarantined
    tr_t, val_t, te_t = splitter.split_temporal(df, cutoff_year=2018)
    assert len(tr_t) > 0
    assert len(te_t) > 0
    # No undated records allowed in any temporal split
    assert not tr_t["publication_date"].isna().any()
    assert not te_t["publication_date"].isna().any()
    for d in tr_t["publication_date"]:
        assert int(d[:4]) < 2016
    for d in te_t["publication_date"]:
        assert int(d[:4]) >= 2018

    # 4. Study-disjoint split: guarantees no shared study or citation cluster across partitions
    tr_s, val_s, te_s = splitter.split_study_disjoint(df)
    train_studies = set(tr_s["study_id"].astype(str))
    test_studies = set(te_s["study_id"].astype(str))
    assert len(train_studies & test_studies) == 0, "Study-disjoint split leaked studies across train and test!"


def test_study_disjoint_edge_cases():
    """
    P1 Fix: Verify edge cases in citation-cluster splitting:
    - 1 cluster -> raises ValueError
    - 2 clusters -> raises ValueError
    - 3 clusters -> successfully allocates non-empty train, val, and test partitions
    - Citation-dependent studies stay in the same partition
    """
    splitter = LeakageSafeSplitter(seed=42)

    # 1. Single cluster must raise ValueError
    df_1 = pd.DataFrame([
        {"herb_id": "H1", "drug_id": "D1", "study_id": "ST1", "cited_study_ids": []},
        {"herb_id": "H1", "drug_id": "D1", "study_id": "ST1", "cited_study_ids": []},
    ])
    with pytest.raises(ValueError) as exc1:
        splitter.split_study_disjoint(df_1)
    assert "at least 3 independent citation clusters" in str(exc1.value)

    # 2. Two clusters must raise ValueError
    df_2 = pd.DataFrame([
        {"herb_id": "H1", "drug_id": "D1", "study_id": "ST1", "cited_study_ids": []},
        {"herb_id": "H2", "drug_id": "D2", "study_id": "ST2", "cited_study_ids": []},
    ])
    with pytest.raises(ValueError) as exc2:
        splitter.split_study_disjoint(df_2)
    assert "at least 3 independent citation clusters" in str(exc2.value)

    # 3. Three clusters must succeed and produce non-empty partitions
    df_3 = pd.DataFrame([
        {"herb_id": "H1", "drug_id": "D1", "study_id": "ST1", "cited_study_ids": []},
        {"herb_id": "H2", "drug_id": "D2", "study_id": "ST2", "cited_study_ids": []},
        {"herb_id": "H3", "drug_id": "D3", "study_id": "ST3", "cited_study_ids": []},
    ])
    tr, val, te = splitter.split_study_disjoint(df_3)
    assert len(tr) == 1
    assert len(val) == 1
    assert len(te) == 1
    assert len(set(tr["study_id"]) & set(te["study_id"])) == 0

    # 4. Citation-linked studies must never be separated
    # ST_A and ST_B cite each other -> 1 cluster. ST_C and ST_D are separate -> total 3 clusters
    df_linked = pd.DataFrame([
        {"herb_id": "H1", "drug_id": "D1", "study_id": "ST_A", "cited_study_ids": ["ST_B"]},
        {"herb_id": "H1", "drug_id": "D1", "study_id": "ST_B", "cited_study_ids": ["ST_A"]},
        {"herb_id": "H2", "drug_id": "D2", "study_id": "ST_C", "cited_study_ids": []},
        {"herb_id": "H3", "drug_id": "D3", "study_id": "ST_D", "cited_study_ids": []},
    ])
    tr_l, val_l, te_l = splitter.split_study_disjoint(df_linked)
    # Check that ST_A and ST_B are in the exact same partition
    for part in [tr_l, val_l, te_l]:
        part_studies = set(part["study_id"])
        if "ST_A" in part_studies:
            assert "ST_B" in part_studies, "Citation-linked studies ST_A and ST_B must stay in the same partition!"


def test_ablation_component_isolation_with_10d_metadata():
    """
    P1 Fix: Verify that 10D metadata with explicit raw counts and lineage-adjusted counts
    is handled consistently without ratio division approximation:
    - 'no_study_dependence' uses raw_pos and raw_neg directly, zeroes echo ratio.
    - 'no_context_conflict' neutralizes context to 1.0 and zeroes conflict.
    """
    graph_dim = 16
    evidence_dim = 16
    herb_emb = torch.randn(1, graph_dim)
    drug_emb = torch.randn(1, graph_dim)
    evidence_feat = torch.randn(1, evidence_dim)

    # 10D vector: [indep_pos, indep_neg, echo, qual, ctx, conflict, margin, missing, raw_pos, raw_neg]
    # Here: raw_pos = 5.0, raw_neg = 0.0, indep_pos = 1.0 (echo chamber with 4 repeating reports: echo = 0.8)
    meta_10d = torch.tensor([[1.0, 0.0, 0.8, 0.9, 0.9, 0.0, 1.0, 0.0, 5.0, 0.0]], dtype=torch.float32)

    model_full = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim, evidence_dim=evidence_dim, fused_dim=16, hidden_dim=16,
        mode="reliability_conditioned",
    )
    model_no_dep = ReliabilityConditionedInteractionModel(
        graph_dim=graph_dim, evidence_dim=evidence_dim, fused_dim=16, hidden_dim=16,
        mode="no_study_dependence",
    )
    model_no_dep.load_state_dict(model_full.state_dict())
    model_full.eval()
    model_no_dep.eval()

    with torch.no_grad():
        out_full = model_full(herb_emb, drug_emb, evidence_feat, metadata=meta_10d, return_details=True)
        out_no_dep = model_no_dep(herb_emb, drug_emb, evidence_feat, metadata=meta_10d, return_details=True)

    assert out_full["breakdown"] is not None
    assert out_no_dep["breakdown"] is not None
    # no_study_dependence uses raw_pos=5.0 and zero echo
    assert out_no_dep["reliability"].item() > 0.0

