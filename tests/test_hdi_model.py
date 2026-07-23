"""
Tests for the full HDI model
"""

import pytest
import torch

from src.models.hdi_model import HDIModel


def test_hdi_model_initialization():
    """Test initializing different variants of the model."""
    
    # Full model
    full_model = HDIModel.full_model(
        gnn_input_dim=16, gnn_hidden_dim=32, gnn_output_dim=16,
        text_output_dim=16
    )
    assert full_model.use_text is True
    assert full_model.use_reliability is True
    assert full_model.reliability_scorer is not None
    assert full_model.cross_attention is not None
    
    # GNN only
    gnn_model = HDIModel.gnn_only(
        gnn_input_dim=16, gnn_hidden_dim=32, gnn_output_dim=16
    )
    assert gnn_model.use_text is False
    assert gnn_model.use_reliability is False
    assert gnn_model.reliability_scorer is None
    assert gnn_model.cross_attention is None
    
    # Unconditioned
    unc_model = HDIModel.unconditioned(
        gnn_input_dim=16, gnn_hidden_dim=32, gnn_output_dim=16,
        text_output_dim=16
    )
    assert unc_model.use_text is True
    assert unc_model.use_reliability is False
    assert unc_model.reliability_scorer is None
    assert unc_model.cross_attention is not None


def test_hdi_model_forward_gnn_only():
    """Test a forward pass through the GNN-only model."""
    model = HDIModel.gnn_only(
        gnn_input_dim=16, gnn_hidden_dim=32, gnn_output_dim=16
    )
    
    num_nodes = 10
    batch_size = 4
    
    node_features = torch.randn(num_nodes, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_type = torch.tensor([0, 1, 0, 1])
    
    source_indices = torch.tensor([0, 1, 2, 3])
    target_indices = torch.tensor([1, 2, 3, 0])
    labels = torch.tensor([1, 0, 1, 0])
    
    out = model(
        node_features=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        source_indices=source_indices,
        target_indices=target_indices,
        labels=labels
    )
    
    assert "logits" in out
    assert "probabilities" in out
    assert "loss" in out
    assert out["logits"].shape == (batch_size,)
