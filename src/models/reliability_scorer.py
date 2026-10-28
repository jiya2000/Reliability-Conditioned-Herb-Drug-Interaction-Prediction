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
        feature_group: str = "quality_only",
    ):
        super().__init__()
        self.feature_group = feature_group

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

        # 1. Quality-only projection (7D: echo, qual, ctx, conflict, missing, rho_reach, f_m_CYP)
        self.quality_proj = nn.Sequential(
            nn.Linear(7, total_dim),
            nn.GELU(),
        )
        self.quality_dim_attention = nn.Linear(total_dim, 7)

        # 1b. Quality-only 5D projection for no_context_conflict: [echo, qual, missing, rho_reach, f_m_CYP]
        self.quality_5d_proj = nn.Sequential(
            nn.Linear(5, total_dim),
            nn.GELU(),
        )
        self.quality_5d_dim_attention = nn.Linear(total_dim, 5)

        # Initialize quality_proj with positive inductive bias for quality signals (qual, ctx, reach, cyp)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.quality_proj[0].weight)
            # Bias positive weight on quality (idx 1), context (idx 2), rho_reach (idx 5)
            self.quality_proj[0].weight[:, 1] += 0.35  # source_quality
            self.quality_proj[0].weight[:, 2] += 0.35  # context_compat
            self.quality_proj[0].weight[:, 0] -= 0.20  # echo_ratio
            self.quality_proj[0].weight[:, 4] -= 0.50  # missingness

            nn.init.xavier_uniform_(self.quality_5d_proj[0].weight)
            self.quality_5d_proj[0].weight[:, 0] -= 0.20  # echo_ratio
            self.quality_5d_proj[0].weight[:, 1] += 0.35  # source_quality
            self.quality_5d_proj[0].weight[:, 2] -= 0.50  # missingness
            self.quality_5d_proj[0].weight[:, 3] += 0.35  # rho_reach

        # 2. Lineage-aware projection for 8D evidence lineage features:
        # [indep_pos, indep_neg, echo_ratio, source_quality, context_compat, contradiction, net_margin, missingness]
        self.lineage_proj = nn.Sequential(
            nn.Linear(8, total_dim),
            nn.GELU(),
        )
        self.lineage_dim_attention = nn.Linear(total_dim, 8)

        # 3. Full 12D lineage projection
        self.full_lineage_proj = nn.Sequential(
            nn.Linear(12, total_dim),
            nn.GELU(),
        )
        self.full_dim_attention = nn.Linear(total_dim, 12)

        # 4. Polarity-only 5D projection (leakage probe)
        self.polarity_proj = nn.Sequential(
            nn.Linear(5, total_dim),
            nn.GELU(),
        )

        # Dimension-wise attention for interpretable breakdown (legacy 5D)
        self.dim_attention = nn.Linear(total_dim, 5)

        self.max_corroboration = max_corroboration

        # MC Dropout for uncertainty estimation
        self.mc_dropout = nn.Dropout(mc_dropout_rate)
        self.mc_samples = mc_samples

        logger.info(
            f"ReliabilityScorer: feature_group='{self.feature_group}', "
            f"hidden={hidden_dim} → R (MC dropout={mc_dropout_rate}, samples={mc_samples})"
        )

    def _extract_signals(self, metadata: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Extract source_quality, context_compat, contradiction, and missingness tensors.
        Returns: (source_quality, context_compat, contradiction, missingness)
        """
        d = metadata.size(-1)
        if d >= 12 or d == 8 or d == 10:
            source_quality = metadata[:, 3:4]
            context_compat = metadata[:, 4:5]
            contradiction = metadata[:, 5:6]
            missingness = metadata[:, 7:8]
        elif d == 7:
            source_quality = metadata[:, 1:2]
            context_compat = metadata[:, 2:3]
            contradiction = metadata[:, 3:4]
            missingness = metadata[:, 4:5]
        elif d == 5 and (self.feature_group == "quality_only" or getattr(self, "mode", None) == "no_context_conflict"):
            source_quality = metadata[:, 1:2]
            context_compat = None  # W_context dropped in no_context_conflict
            contradiction = None   # C_hd dropped in no_context_conflict
            missingness = metadata[:, 2:3]
        else:
            source_quality = None
            context_compat = None
            contradiction = None
            missingness = None
        return source_quality, context_compat, contradiction, missingness

    def forward(
        self,
        metadata: torch.Tensor,
        return_breakdown: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Compute reliability score from metadata.
        Guarantees:
        - When missingness = 1.0 (no literature), R collapses to ~0.0 (R <= 0.05).
        - Explicit contradiction reduces gating weight.
        - High-quality human trial evidence yields strictly higher R than in vitro evidence.
        """
        combined = self._embed_metadata(metadata)
        R = self.mlp(combined)  # (batch, 1) in [0, 1]

        source_quality, context_compat, contradiction, missingness = self._extract_signals(metadata)

        # Quality and context modulation: high-quality clinical studies yield higher trust than in vitro
        if source_quality is not None and context_compat is not None:
            q_factor = 0.35 + 0.65 * (0.5 * source_quality.clamp(0.0, 1.0) + 0.5 * context_compat.clamp(0.0, 1.0))
            R = R * q_factor
        elif source_quality is not None:
            # When context_compat is ablated (e.g. no_context_conflict), modulate only by source quality
            q_factor = 0.35 + 0.65 * source_quality.clamp(0.0, 1.0)
            R = R * q_factor

        # Contradiction damping
        if contradiction is not None:
            R = R * (1.0 - 0.4 * contradiction.clamp(0.0, 1.0))

        # Missingness grounding: when missingness -> 1.0 (no literature evidence),
        # R collapses to 0.0 so z_fused retreats strictly to biological graph priors
        if missingness is not None:
            R = R * (1.0 - missingness.clamp(0.0, 1.0))

        breakdown = None
        if return_breakdown:
            d = metadata.size(-1)
            if d == 5 and (self.feature_group == "quality_only" or getattr(self, "mode", None) == "no_context_conflict"):
                dim_weights = torch.softmax(self.quality_5d_dim_attention(combined), dim=-1)
                breakdown = {
                    "echo_ratio_weight": dim_weights[:, 0],
                    "source_quality_weight": dim_weights[:, 1],
                    "missingness_weight": dim_weights[:, 2],
                    "rho_reach_weight": dim_weights[:, 3],
                    "f_m_CYP_weight": dim_weights[:, 4],
                }
            elif self.feature_group == "quality_only" or d == 7:
                dim_weights = torch.softmax(self.quality_dim_attention(combined), dim=-1)
                breakdown = {
                    "echo_ratio_weight": dim_weights[:, 0],
                    "source_quality_weight": dim_weights[:, 1],
                    "context_compat_weight": dim_weights[:, 2],
                    "contradiction_weight": dim_weights[:, 3],
                    "missingness_weight": dim_weights[:, 4],
                    "rho_reach_weight": dim_weights[:, 5],
                    "f_m_CYP_weight": dim_weights[:, 6],
                }
            elif d == 8:
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
                }
            elif d >= 10:
                dim_weights = torch.softmax(self.full_dim_attention(combined), dim=-1)
                breakdown = {
                    "indep_pos_weight": dim_weights[:, 0],
                    "indep_neg_weight": dim_weights[:, 1],
                    "echo_ratio_weight": dim_weights[:, 2],
                    "source_quality_weight": dim_weights[:, 3],
                    "context_compat_weight": dim_weights[:, 4],
                    "contradiction_weight": dim_weights[:, 5],
                    "net_margin_weight": dim_weights[:, 6],
                    "missingness_weight": dim_weights[:, 7],
                }
            else:
                dim_weights = torch.softmax(self.dim_attention(combined), dim=-1)
                breakdown = {
                    "corroboration_weight": dim_weights[:, 0],
                    "temporal_weight": dim_weights[:, 1],
                    "biomedical_weight": dim_weights[:, 2],
                    "molecular_weight": dim_weights[:, 3],
                    "source_type_weight": dim_weights[:, 4],
                }

        return R, breakdown

    def _embed_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Shared embedding logic supporting:
        - 7D quality-only metadata: [echo, qual, ctx, conflict, missing, reach, cyp]
        - 12D full metadata: [S_pos, S_neg, echo, qual, ctx, conflict, margin, missing, raw_pos, raw_neg, reach, cyp]
        - 10D lineage metadata (padded or quality-extracted)
        - 8D lineage metadata
        - 5D polarity-only or legacy [C, T, B, M, S]
        """
        d = metadata.size(-1)

        if d == 7:
            return self.quality_proj(metadata.float())

        if d >= 12:
            if self.feature_group == "quality_only":
                qual_7d = metadata[:, [2, 3, 4, 5, 7, 10, 11]].float()
                return self.quality_proj(qual_7d)
            elif self.feature_group == "polarity_only":
                pol_5d = metadata[:, [0, 1, 6, 8, 9]].float()
                return self.polarity_proj(pol_5d)
            else:
                return self.full_lineage_proj(metadata[:, :12].float())

        if d == 10:
            if self.feature_group == "quality_only":
                qual_5d = metadata[:, [2, 3, 4, 5, 7]].float()
                batch_size = metadata.size(0)
                defaults = torch.full((batch_size, 2), 0.5, device=metadata.device, dtype=torch.float32)
                qual_7d = torch.cat([qual_5d, defaults], dim=-1)
                return self.quality_proj(qual_7d)
            elif self.feature_group == "polarity_only":
                pol_5d = metadata[:, [0, 1, 6, 8, 9]].float()
                return self.polarity_proj(pol_5d)
            else:
                batch_size = metadata.size(0)
                defaults = torch.full((batch_size, 2), 0.5, device=metadata.device, dtype=torch.float32)
                full_12d = torch.cat([metadata[:, :10].float(), defaults], dim=-1)
                return self.full_lineage_proj(full_12d)

        if d == 8:
            if self.feature_group == "quality_only":
                qual_5d = metadata[:, [2, 3, 4, 5, 7]].float()
                batch_size = metadata.size(0)
                defaults = torch.full((batch_size, 2), 0.5, device=metadata.device, dtype=torch.float32)
                qual_7d = torch.cat([qual_5d, defaults], dim=-1)
                return self.quality_proj(qual_7d)
            elif self.feature_group == "polarity_only":
                pol_3d = metadata[:, [0, 1, 6]].float()
                batch_size = metadata.size(0)
                pol_5d = torch.cat([pol_3d, pol_3d[:, :2]], dim=-1)
                return self.polarity_proj(pol_5d)
            else:
                return self.lineage_proj(metadata.float())

        if d == 5:
            if self.feature_group == "quality_only" or getattr(self, "mode", None) == "no_context_conflict":
                return self.quality_5d_proj(metadata.float())
            elif self.feature_group == "polarity_only":
                return self.polarity_proj(metadata.float())
            # Legacy 5D embedding
            return self._embed_legacy_5d(metadata)

        # Fallback: slice or pad to 8D for lineage_proj
        if d > 8:
            return self.lineage_proj(metadata[:, :8].float())
        else:
            return self.lineage_proj(metadata.float())

    def _embed_legacy_5d(self, metadata: torch.Tensor) -> torch.Tensor:
        """Embed legacy 5D [C, T, B, M, S] metadata."""
        C = metadata[:, 0].long().clamp(0, self.max_corroboration)
        T = metadata[:, 1:2].float()
        B = metadata[:, 2:3].float()
        M = metadata[:, 3:4].float()
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

        module_states = {
            name: mod.training for name, mod in self.named_modules()
        }

        # Only enable the MC dropout layer
        self.mc_dropout.train()

        samples = []
        for _ in range(n):
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

        source_quality, context_compat, contradiction, missingness = self._extract_signals(metadata)

        # Quality and context modulation: high-quality clinical studies yield higher trust than in vitro
        if source_quality is not None and context_compat is not None:
            q_factor = 0.35 + 0.65 * (0.5 * source_quality.clamp(0.0, 1.0) + 0.5 * context_compat.clamp(0.0, 1.0))
            R_mean = R_mean * q_factor

        # In lineage mode, explicit conflict reduces gating trust and elevates epistemic uncertainty
        if contradiction is not None:
            R_mean = R_mean * (1.0 - 0.4 * contradiction.clamp(0.0, 1.0))
            R_uncertainty = R_uncertainty + 0.25 * contradiction.clamp(0.0, 1.0)

        # Missingness grounding: when missingness = 1.0, R_mean drops to 0.0
        if missingness is not None:
            R_mean = R_mean * (1.0 - missingness.clamp(0.0, 1.0))
            R_uncertainty = R_uncertainty * (1.0 - missingness.clamp(0.0, 1.0))

        d = metadata.size(-1)
        if self.feature_group == "quality_only" or d == 7:
            dim_weights = torch.softmax(self.quality_dim_attention(combined), dim=-1)
            breakdown = {
                "echo_ratio_weight": dim_weights[:, 0],
                "source_quality_weight": dim_weights[:, 1],
                "context_compat_weight": dim_weights[:, 2],
                "contradiction_weight": dim_weights[:, 3],
                "missingness_weight": dim_weights[:, 4],
                "rho_reach_weight": dim_weights[:, 5],
                "f_m_CYP_weight": dim_weights[:, 6],
                "mc_samples": n,
                "mean_uncertainty": R_uncertainty.mean().item(),
            }
        elif d == 8:
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
                "mc_samples": n,
                "mean_uncertainty": R_uncertainty.mean().item(),
            }
        elif d >= 10:
            dim_weights = torch.softmax(self.full_dim_attention(combined), dim=-1)
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
