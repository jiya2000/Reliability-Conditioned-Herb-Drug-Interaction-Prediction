"""
End-to-end integration tests for all 3 novel contributions:
1. Citation DAG Echo Attenuation (CDA)
2. Pharmacokinetic Context Compatibility (PCC)
3. Conformal Tri-State Abstention Engine (CTAE)
"""

import numpy as np
import pytest
import torch
from src.models.hdi_model import HDIModel


def test_full_novelty_pipeline():
    """Verify HDIModel with CDA, PCC, and CTAE all active simultaneously."""
    batch_size = 4
    num_nodes = 20
    hidden_dim = 64

    model = HDIModel(
        gnn_input_dim=hidden_dim,
        gnn_hidden_dim=hidden_dim,
        gnn_output_dim=hidden_dim,
        num_relations=3,
        gnn_layers=2,
        use_text=False,  # Test without heavy BERT download
        use_reliability=True,
        enable_echo_attenuation=True,
        enable_pk_context=True,
        conformal_epsilon=0.10,
    )

    node_features = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    edge_type = torch.randint(0, 3, (40,))
    source_indices = torch.tensor([0, 1, 2, 3])
    target_indices = torch.tensor([4, 5, 6, 7])

    # 8D Evidence Lineage metadata:
    # [indep_pos, indep_neg, echo_ratio, source_quality, context_compat, contradiction, net_margin, missingness]
    metadata = torch.tensor([
        [5.0, 0.0, 0.80, 0.90, 0.95, 0.05, 5.0, 0.0],  # Strong positive with high echo
        [1.0, 4.0, 0.10, 0.85, 0.80, 0.10, -3.0, 0.0], # Clear negative
        [3.0, 3.0, 0.20, 0.80, 0.70, 0.85, 0.0, 0.0],  # Severe contradiction (conflict = 0.85)
        [2.0, 0.0, 0.30, 0.60, 0.50, 0.20, 2.0, 0.1],  # Moderate evidence
    ], dtype=torch.float32)

    # 1. Forward pass
    output = model(
        node_features=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        source_indices=source_indices,
        target_indices=target_indices,
        metadata=metadata,
    )

    assert "probabilities" in output
    assert "reliability_scores" in output
    assert output["probabilities"].shape == (batch_size,)

    # 2. Calibrate conformal engine on small calibration set
    cal_probs = np.array([0.92, 0.85, 0.15, 0.08, 0.78, 0.22, 0.90, 0.10])
    cal_labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    q_hat = model.calibrate_conformal(cal_probs, cal_labels, epsilon=0.10)
    assert 0.0 < q_hat < 1.0

    # 3. Predict with conformal abstention
    res = model.predict_with_conformal_abstention(
        node_features=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        source_indices=source_indices,
        target_indices=target_indices,
        metadata=metadata,
    )

    assert "conformal_decisions" in res
    assert "predicted_labels" in res
    assert "abstention_mask" in res
    assert len(res["conformal_decisions"]) == batch_size

    # The 3rd sample has contradiction = 0.85 (> 0.60 threshold)
    # It MUST be forced to abstain!
    decision_3 = res["conformal_decisions"][2]
    assert decision_3.abstain
    assert decision_3.predicted_label == -1
    assert "contradiction" in decision_3.abstention_reason

    # Count parameters check
    param_counts = model.count_parameters()
    assert "echo_attenuator" in param_counts
    assert "context_scorer" in param_counts
    assert param_counts["echo_attenuator"] > 0
    assert param_counts["context_scorer"] > 0
