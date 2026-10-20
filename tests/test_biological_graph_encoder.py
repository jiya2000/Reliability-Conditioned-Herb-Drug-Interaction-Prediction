"""
Unit tests for HeterogeneousBiologicalGraphEncoder.
Tests tripartite graph construction, GAT message passing, entity embedding lookup,
and gradient flow.
"""

import pytest
import torch
from src.features.biological_graph_encoder import HeterogeneousBiologicalGraphEncoder


def test_heterogeneous_biological_graph_encoder():
    encoder = HeterogeneousBiologicalGraphEncoder(hidden_dim=16, num_heads=2)

    # 1. Forward pass for all nodes
    all_h, all_d = encoder.compute_all_node_embeddings()
    assert all_h.dim() == 2
    assert all_h.size(1) == 16
    assert all_d.dim() == 2
    assert all_d.size(1) == 16

    # Verify unit normalization
    h_norms = torch.norm(all_h, p=2, dim=-1)
    d_norms = torch.norm(all_d, p=2, dim=-1)
    assert torch.allclose(h_norms, torch.ones_like(h_norms), atol=1e-4)
    assert torch.allclose(d_norms, torch.ones_like(d_norms), atol=1e-4)

    # 2. Batch entity encoding
    herb_ids = ["HERB_HYPERICUM", "HERB_GINKGO", "HERB_CURCUMA"]
    drug_ids = ["DRUG_CYCLOSPORINE", "DRUG_WARFARIN", "DRUG_TACROLIMUS"]

    batch_h, batch_d, interaction = encoder(herb_ids, drug_ids)
    assert batch_h.shape == (3, 16)
    assert batch_d.shape == (3, 16)
    assert interaction.shape == (3, 16)

    # 3. Gradient backpropagation check (end-to-end differentiability)
    loss = (interaction.sum()) ** 2
    loss.backward()

    # Verify gradients exist in GAT parameters
    assert encoder.gat1_t_to_h.lin_src.weight.grad is not None
    assert encoder.herb_encoder[0].weight.grad is not None
