"""
Tests for the core Reliability-Conditioned Cross-Attention
"""

import pytest
import torch

from src.models.cross_attention import ReliabilityConditionedCrossAttention


def test_cross_attention_multiplicative():
    """Test the default multiplicative gating mode."""
    batch_size = 4
    hidden_dim = 16
    heads = 2
    
    layer = ReliabilityConditionedCrossAttention(
        hidden_dim=hidden_dim, 
        num_heads=heads, 
        gating_mode="multiplicative"
    )
    
    mol_emb = torch.randn(batch_size, hidden_dim)
    text_emb = torch.randn(batch_size, hidden_dim)
    
    # R = 1 (full trust)
    r_ones = torch.ones(batch_size, 1)
    out_ones, attn_ones = layer(mol_emb, text_emb, r_ones, return_attention=True)
    
    # R = 0 (no trust)
    r_zeros = torch.zeros(batch_size, 1)
    out_zeros, attn_zeros = layer(mol_emb, text_emb, r_zeros, return_attention=True)
    
    assert out_ones.shape == (batch_size, hidden_dim)
    assert out_zeros.shape == (batch_size, hidden_dim)
    
    # When R=0, attention weights should be 0 (multiplicative)
    assert torch.allclose(attn_zeros, torch.zeros_like(attn_zeros))
    # When R=1, attention weights should be > 0
    assert not torch.allclose(attn_ones, torch.zeros_like(attn_ones))


def test_cross_attention_unconditioned():
    """Test the unconditioned ablation baseline."""
    batch_size = 2
    hidden_dim = 16
    
    layer = ReliabilityConditionedCrossAttention(hidden_dim=hidden_dim)
    mol_emb = torch.randn(batch_size, hidden_dim)
    text_emb = torch.randn(batch_size, hidden_dim)
    
    out = layer.forward_unconditioned(mol_emb, text_emb)
    assert out.shape == (batch_size, hidden_dim)
