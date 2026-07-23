"""
Reliability Scorer

Computes the scalar reliability score R ∈ [0, 1] from the five
candidate metadata dimensions (C, T, B, M, S).

R is NOT directly supervised — it is learned end-to-end by
backpropagating the link-prediction loss on DrugBank anchor edges
through the full pipeline (extraction → R → cross-attention → prediction).

This is a critical component of the core invention: R dynamically
gates the cross-attention weights, controlling how much the model
trusts the textual evidence for each candidate interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class ReliabilityScorer(nn.Module):
    """
    Computes reliability score R from candidate metadata.

    Architecture:
    1. Embed each metadata dimension separately (learned embeddings)
    2. Concatenate embedded dimensions
    3. MLP → sigmoid → R ∈ [0, 1]

    The five metadata dimensions:
    - C (corroboration): int → embedded (number of confirming sources)
    - T (temporal recency): float ∈ [0, 1]
    - B (biomedical quality): float ∈ [0, 1]
    - M (molecular plausibility): float ∈ [0, 1]
    - S (source type): int → embedded (categorical)

    Usage:
        scorer = ReliabilityScorer()
        metadata = torch.tensor([[3, 0.8, 0.9, 0.6, 2]])  # (batch, 5)
        R, breakdown = scorer(metadata)  # R: (batch, 1), breakdown: dict
    """

    SOURCE_TYPES = [
        "unknown", "clinical_trial", "peer_reviewed", "textbook",
        "case_report", "health_forum", "social_media", "traditional_medicine",
    ]

    def __init__(
        self,
        corroboration_embed_dim: int = 16,
        temporal_dim: int = 16,
        biomedical_dim: int = 16,
        molecular_dim: int = 16,
        source_type_embed_dim: int = 16,
        hidden_dim: int = 64,
        max_corroboration: int = 50,
        num_source_types: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Dimension-specific embeddings/projections
        # C: Corroboration count → embedding
        self.corroboration_embed = nn.Embedding(
            max_corroboration + 1, corroboration_embed_dim
        )

        # T: Temporal recency (scalar) → projected
        self.temporal_proj = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.GELU(),
        )

        # B: Biomedical quality (scalar) → projected
        self.biomedical_proj = nn.Sequential(
            nn.Linear(1, biomedical_dim),
            nn.GELU(),
        )

        # M: Molecular plausibility (scalar) → projected
        self.molecular_proj = nn.Sequential(
            nn.Linear(1, molecular_dim),
            nn.GELU(),
        )

        # S: Source type → embedding
        self.source_type_embed = nn.Embedding(
            num_source_types, source_type_embed_dim
        )

        # Combined dimension
        total_dim = (
            corroboration_embed_dim
            + temporal_dim
            + biomedical_dim
            + molecular_dim
            + source_type_embed_dim
        )

        # MLP to produce R
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # R ∈ [0, 1]
        )

        # Dimension-wise attention for interpretable breakdown
        self.dim_attention = nn.Linear(total_dim, 5)

        self.max_corroboration = max_corroboration

        logger.info(
            f"ReliabilityScorer: C={corroboration_embed_dim}, "
            f"T={temporal_dim}, B={biomedical_dim}, "
            f"M={molecular_dim}, S={source_type_embed_dim} "
            f"→ hidden={hidden_dim} → R"
        )

    def forward(
        self,
        metadata: torch.Tensor,
        return_breakdown: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Compute reliability score from metadata.

        Args:
            metadata: (batch, 5) tensor with columns [C, T, B, M, S]
                C: int (corroboration count)
                T: float (temporal recency)
                B: float (biomedical quality)
                M: float (molecular plausibility)
                S: int (source type code)
            return_breakdown: If True, return per-dimension contributions

        Returns:
            R: (batch, 1) reliability score
            breakdown: Optional dict with per-dimension contributions
        """
        batch_size = metadata.size(0)

        # Extract individual dimensions
        C = metadata[:, 0].long().clamp(0, self.max_corroboration)  # int
        T = metadata[:, 1:2]  # (batch, 1) float
        B = metadata[:, 2:3]  # (batch, 1) float
        M = metadata[:, 3:4]  # (batch, 1) float
        S = metadata[:, 4].long().clamp(0, 7)  # int

        # Embed/project each dimension
        c_emb = self.corroboration_embed(C)  # (batch, c_dim)
        t_emb = self.temporal_proj(T)  # (batch, t_dim)
        b_emb = self.biomedical_proj(B)  # (batch, b_dim)
        m_emb = self.molecular_proj(M)  # (batch, m_dim)
        s_emb = self.source_type_embed(S)  # (batch, s_dim)

        # Concatenate all dimension embeddings
        combined = torch.cat([c_emb, t_emb, b_emb, m_emb, s_emb], dim=-1)

        # Compute R
        R = self.mlp(combined)  # (batch, 1)

        breakdown = None
        if return_breakdown:
            # Compute attention over dimensions for interpretability
            dim_weights = torch.softmax(
                self.dim_attention(combined), dim=-1
            )  # (batch, 5)

            breakdown = {
                "corroboration_weight": dim_weights[:, 0],
                "temporal_weight": dim_weights[:, 1],
                "biomedical_weight": dim_weights[:, 2],
                "molecular_weight": dim_weights[:, 3],
                "source_type_weight": dim_weights[:, 4],
                "raw_values": {
                    "C": metadata[:, 0],
                    "T": metadata[:, 1],
                    "B": metadata[:, 2],
                    "M": metadata[:, 3],
                    "S": metadata[:, 4],
                },
            }

        return R, breakdown
