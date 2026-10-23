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

★ NOVEL CONTRIBUTION 2: Uncertainty Quantification ★
Supports Monte Carlo dropout at inference time to produce
(R_mean, R_uncertainty) — enabling uncertainty-aware gating
where the system behaves conservatively under high epistemic uncertainty.
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
        mc_dropout_rate: float = 0.15,
        mc_samples: int = 10,
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

        # Lineage-aware projection for 8D evidence lineage features:
        # [indep_pos, indep_neg, echo_ratio, source_quality, context_compat, contradiction, net_margin, missingness]
        self.lineage_proj = nn.Sequential(
            nn.Linear(8, total_dim),
            nn.GELU(),
        )
        self.lineage_dim_attention = nn.Linear(total_dim, 8)

        # Dimension-wise attention for interpretable breakdown
        self.dim_attention = nn.Linear(total_dim, 5)

        self.max_corroboration = max_corroboration

        # MC Dropout for uncertainty estimation
        self.mc_dropout = nn.Dropout(mc_dropout_rate)
        self.mc_samples = mc_samples

        logger.info(
            f"ReliabilityScorer: C={corroboration_embed_dim}, "
            f"T={temporal_dim}, B={biomedical_dim}, "
            f"M={molecular_dim}, S={source_type_embed_dim} "
            f"→ hidden={hidden_dim} → R (supports 5D legacy & 8D evidence lineage) "
            f"(MC dropout={mc_dropout_rate}, samples={mc_samples})"
        )


    def forward(
        self,
        metadata: torch.Tensor,
        return_breakdown: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Compute reliability score from metadata.

        Args:
            metadata: Tensor of shape (batch, 5) or (batch, 8).
                If (batch, 5) [Legacy]:
                    [C, T, B, M, S]
                If (batch, 8) [Evidence Lineage]:
                    [indep_pos, indep_neg, echo_ratio, source_quality,
                     context_compat, contradiction, net_margin, missingness]
            return_breakdown: If True, return per-dimension contributions

        Returns:
            R: (batch, 1) reliability score
            breakdown: Optional dict with per-dimension contributions
        """
        combined = self._embed_metadata(metadata)
        R = self.mlp(combined)  # (batch, 1)

        # In lineage mode, explicit conflict reduces gating trust
        if metadata.size(-1) >= 8:
            contradiction = metadata[:, 5:6]  # (batch, 1) in [0, 1]
            R = R * (1.0 - 0.4 * contradiction)

        breakdown = None
        if return_breakdown:
            if metadata.size(-1) >= 8:
                dim_weights = torch.softmax(self.lineage_dim_attention(combined), dim=-1)
                breakdown = {
                    "indep_pos_weight": dim_weights[:, 0],
                    "indep_neg_weight": dim_weights[:, 1],
                    "echo_ratio_weight": dim_weights[:, 2],
                    "source_quality_weight": dim_weights[:, 3],
                    "context_compat_weight": dim_weights[:, 4],
                    "contradiction_weight": dim_weights[:, 5],
                    "net_margin_weight": dim_weights[:, 6],
                    "missingness_weight": dim_weights[:, 7],
                    "raw_values": {
                        "indep_pos": metadata[:, 0],
                        "indep_neg": metadata[:, 1],
                        "echo_ratio": metadata[:, 2],
                        "source_quality": metadata[:, 3],
                        "context_compat": metadata[:, 4],
                        "contradiction": metadata[:, 5],
                        "net_margin": metadata[:, 6],
                        "missingness": metadata[:, 7],
                    },
                }
            else:
                dim_weights = torch.softmax(self.dim_attention(combined), dim=-1)
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

    def _embed_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Shared embedding logic supporting both 8D evidence lineage and 5D legacy metadata.
        """
        if metadata.size(-1) >= 8:
            return self.lineage_proj(metadata[:, :8].float())

        # Legacy 5D embedding
        C = metadata[:, 0].long().clamp(0, self.max_corroboration)
        T = metadata[:, 1:2]
        B = metadata[:, 2:3]
        M = metadata[:, 3:4]
        S = metadata[:, 4].long().clamp(0, 7)

        c_emb = self.corroboration_embed(C)
        t_emb = self.temporal_proj(T)
        b_emb = self.biomedical_proj(B)
        m_emb = self.molecular_proj(M)
        s_emb = self.source_type_embed(S)

        return torch.cat([c_emb, t_emb, b_emb, m_emb, s_emb], dim=-1)


    def forward_with_uncertainty(
        self,
        metadata: torch.Tensor,
        n_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """
        ★ NOVEL: MC Dropout Uncertainty Estimation ★

        Run multiple stochastic forward passes to estimate:
        - R_mean: expected reliability score
        - R_uncertainty: epistemic uncertainty (std dev of MC samples)

        Args:
            metadata: (batch, 5) tensor with columns [C, T, B, M, S]
            n_samples: number of MC samples (default: self.mc_samples)

        Returns:
            R_mean: (batch, 1) mean reliability score
            R_uncertainty: (batch, 1) uncertainty estimate (std dev)
            breakdown: dict with per-dimension contributions
        """
        n = n_samples or self.mc_samples

        # Embed once (deterministic part)
        combined = self._embed_metadata(metadata)

        # ★ FIX: Save per-module training states instead of calling
        # self.train() which would enable dropout on ALL submodules
        # (including self.mlp's internal Dropout layers), creating
        # uncontrolled double-stochastic behavior.
        module_states = {
            name: mod.training for name, mod in self.named_modules()
        }

        # Only enable the MC dropout layer
        self.mc_dropout.train()

        samples = []
        for _ in range(n):
            # Apply MC dropout to the combined embedding
            mc_combined = self.mc_dropout(combined)
            R_sample = self.mlp(mc_combined)
            samples.append(R_sample)

        # Restore original training state for each module
        for name, mod in self.named_modules():
            if name in module_states:
                mod.train(module_states[name])

        # Stack: (n_samples, batch, 1)
        R_stack = torch.stack(samples, dim=0)

        # Compute mean and std
        R_mean = R_stack.mean(dim=0)           # (batch, 1)
        R_uncertainty = R_stack.std(dim=0)      # (batch, 1)

        # In lineage mode, explicit conflict reduces gating trust
        if metadata.size(-1) == 8:
            contradiction = metadata[:, 5:6]
            R_mean = R_mean * (1.0 - 0.4 * contradiction)
            dim_weights = torch.softmax(self.lineage_dim_attention(combined), dim=-1)
            # Epistemic uncertainty is also elevated when literature conflict is present
            R_uncertainty = R_uncertainty + 0.25 * contradiction
            breakdown = {
                "indep_pos_weight": dim_weights[:, 0],
                "indep_neg_weight": dim_weights[:, 1],
                "echo_ratio_weight": dim_weights[:, 2],
                "source_quality_weight": dim_weights[:, 3],
                "context_compat_weight": dim_weights[:, 4],
                "contradiction_weight": dim_weights[:, 5],
                "net_margin_weight": dim_weights[:, 6],
                "missingness_weight": dim_weights[:, 7],
                "mc_samples": n,
                "mean_uncertainty": R_uncertainty.mean().item(),
            }
        else:
            dim_weights = torch.softmax(self.dim_attention(combined), dim=-1)
            breakdown = {
                "corroboration_weight": dim_weights[:, 0],
                "temporal_weight": dim_weights[:, 1],
                "biomedical_weight": dim_weights[:, 2],
                "molecular_weight": dim_weights[:, 3],
                "source_type_weight": dim_weights[:, 4],
                "mc_samples": n,
                "mean_uncertainty": R_uncertainty.mean().item(),
            }

        return R_mean, R_uncertainty, breakdown
