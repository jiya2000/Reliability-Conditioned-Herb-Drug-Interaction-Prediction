"""
Reliability-Conditioned Cross-Attention

★ CORE INVENTION ★

This module implements the central innovation of the system: using
the learned reliability score R as a dynamic gate on the cross-attention
weights that fuse heterogeneous textual embeddings with molecular
graph embeddings.

The key insight: R is NOT a post-hoc filter (like NeuroGRIP), NOT a
voting weight in closed-set fusion (like BELIEF), and NOT an intra-modal
alignment signal (like DDI-AttendNet). It is a direct, dynamic gate on
the cross-attention weights themselves, resolving the multimodal
alignment problem between fixed-dimension molecular representations
and noisy, variable-length, code-mixed textual representations.

Three gating modes are supported:
1. Multiplicative: attention_weights *= R (default, simplest)
2. Additive: attention_weights += R * learned_bias
3. Learned gate: g = σ(W·[R, attention_weights]) → gated_weights
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ReliabilityConditionedCrossAttention(nn.Module):
    """
    Cross-attention layer where the reliability score R dynamically
    gates the text-to-molecule attention weights.

    This resolves a specific technical problem: fusing a fixed-dimension
    molecular graph embedding with a noisy, variable-length, code-mixed
    textual embedding, where the relative trust in the textual signal
    varies per-instance and must be learned.

    Architecture:
        Q = W_q · molecular_embedding  (from GNN)
        K = W_k · text_embedding       (from text encoder)
        V = W_v · text_embedding

        raw_attention = softmax(Q · K^T / √d)
        gated_attention = R ⊙ raw_attention     ← Core gating step
        output = gated_attention · V

    Usage:
        cross_attn = ReliabilityConditionedCrossAttention(
            hidden_dim=128, num_heads=4
        )
        fused = cross_attn(
            molecular_embedding,  # (batch, mol_dim)
            text_embedding,       # (batch, text_dim)
            reliability_score,    # (batch, 1)
        )
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        gating_mode: str = "multiplicative",
        temperature: float = 1.0,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gating_mode = gating_mode
        self.temperature = temperature
        self.scale = math.sqrt(self.head_dim)

        # Query projection (from molecular embeddings)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)

        # Key and Value projections (from text embeddings)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Gating components
        if gating_mode == "additive":
            self.gate_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        elif gating_mode == "learned_gate":
            self.gate_net = nn.Sequential(
                nn.Linear(num_heads + 1, num_heads),
                nn.Sigmoid(),
            )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        logger.info(
            f"ReliabilityConditionedCrossAttention: "
            f"dim={hidden_dim}, heads={num_heads}, "
            f"gating={gating_mode}, temp={temperature}"
        )

    def _apply_reliability_gating(
        self,
        attention_weights: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reliability-conditioned gating to attention weights.

        This is the core inventive step — R modulates how much the model
        attends to textual evidence when fusing with molecular embeddings.

        Args:
            attention_weights: (batch, num_heads, 1, 1) attention weights
            R: (batch, 1) reliability score ∈ [0, 1]

        Returns:
            gated_weights: (batch, num_heads, 1, 1) gated attention weights
        """
        # Reshape R for broadcasting: (batch, 1, 1, 1)
        R_expanded = R.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, 1)

        if self.gating_mode == "multiplicative":
            # Simplest and most interpretable:
            # When R → 0, text attention is suppressed
            # When R → 1, text attention is fully used
            gated = attention_weights * R_expanded

        elif self.gating_mode == "additive":
            # R shifts the attention distribution
            gated = attention_weights + R_expanded * self.gate_bias

        elif self.gating_mode == "learned_gate":
            # Learn a per-head gating function from R and attention weights
            batch_size = attention_weights.size(0)
            # Average attention per head
            attn_per_head = attention_weights.mean(dim=(-2, -1))  # (batch, heads)
            gate_input = torch.cat(
                [attn_per_head, R], dim=-1
            )  # (batch, heads+1)
            gate = self.gate_net(gate_input)  # (batch, heads)
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # (batch, heads, 1, 1)
            gated = attention_weights * gate

        else:
            raise ValueError(f"Unknown gating mode: {self.gating_mode}")

        return gated

    def forward(
        self,
        molecular_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        reliability_score: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: reliability-conditioned cross-attention fusion.

        Args:
            molecular_embedding: (batch, hidden_dim) from GNN encoder
            text_embedding: (batch, hidden_dim) from text encoder
            reliability_score: (batch, 1) R from reliability scorer
            return_attention: If True, return attention weights for analysis

        Returns:
            fused_embedding: (batch, hidden_dim) fused representation
            attention_weights: Optional (batch, num_heads, 1, 1)
        """
        batch_size = molecular_embedding.size(0)

        # Project to Q, K, V
        # Q from molecular (the "query" — what molecular info seeks from text)
        Q = self.W_q(molecular_embedding)  # (batch, hidden_dim)
        # K, V from text (the "knowledge" and "value" from textual evidence)
        K = self.W_k(text_embedding)  # (batch, hidden_dim)
        V = self.W_v(text_embedding)  # (batch, hidden_dim)

        # Reshape for multi-head attention
        # (batch, hidden_dim) → (batch, num_heads, 1, head_dim)
        Q = Q.view(batch_size, self.num_heads, 1, self.head_dim)
        K = K.view(batch_size, self.num_heads, 1, self.head_dim)
        V = V.view(batch_size, self.num_heads, 1, self.head_dim)

        # Compute attention scores
        # (batch, heads, 1, head_dim) × (batch, heads, head_dim, 1) = (batch, heads, 1, 1)
        raw_attention = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.scale * self.temperature
        )
        attention_weights = torch.sigmoid(raw_attention)

        # ★ CORE GATING STEP ★
        # Apply reliability-conditioned gating
        gated_attention = self._apply_reliability_gating(
            attention_weights, reliability_score
        )

        # Apply attention dropout
        gated_attention = self.attn_dropout(gated_attention)

        # Weighted sum of values
        # (batch, heads, 1, 1) × (batch, heads, 1, head_dim) = (batch, heads, 1, head_dim)
        attended = gated_attention * V

        # Reshape back: (batch, heads, 1, head_dim) → (batch, hidden_dim)
        attended = attended.view(batch_size, self.hidden_dim)

        # Output projection
        output = self.W_o(attended)
        output = self.dropout(output)

        # Residual connection with molecular embedding + layer norm
        fused = self.layer_norm(output + molecular_embedding)

        if return_attention:
            return fused, gated_attention.squeeze(-1).squeeze(-1)
        return fused, None

    def forward_unconditioned(
        self,
        molecular_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass WITHOUT reliability conditioning.

        Used as the ablation baseline — same architecture,
        but R is fixed to 1.0 (no gating).

        This is variant (b) in the Week 12 ablation:
        "unconditioned cross-attention (same architecture, R removed)"
        """
        batch_size = molecular_embedding.size(0)
        R_ones = torch.ones(batch_size, 1, device=molecular_embedding.device)
        fused, _ = self.forward(
            molecular_embedding, text_embedding, R_ones
        )
        return fused
